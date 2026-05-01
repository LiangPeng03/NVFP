import re
import gc
import math
import argparse
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import AutoModelForCausalLM

from .qlinear import QLinear
from .quantizer import Quantizer
from .quant_args import QuantizationOrder
from .quant_ops import pack_fp4_to_uint8, cast_scales_to_eXmY, ScalePrecision
from .accumulate_hessian import accumulate_hessian
from ..transforms.transforms import build_transform, get_transform_matrix
from ..utils.linalg_utils import inv_sym
from ..utils.common_utils import clear_device_cache, to, maybe_first_element
from ..utils.model_utils import InputCollector, ForwardInterrupt, get_attention_layer, get_mlp_layer, get_number_of_rows_and_cols

try:
    import wandb
except ImportError:
    wandb = None


def get_relative_mse_error(q: torch.Tensor, w: torch.Tensor, H: torch.Tensor):
    delta = q - w
    return (delta).mm(H).mul(delta).mean() / (w.mm(H).mul(w).mean() + 1e-6)


class GPTAQ:

    def __init__(
        self,
        layer: nn.Module,
        quantizer: Quantizer,
        quantization_order: str = "default",
        block_size: int = 128,
        rel_damp: float = 1e-2,
        export_quantized_model: str = "",
        alpha: str = "auto",
        rtm_lambda: Optional[float] = None,
    ):
        assert isinstance(layer, (nn.Linear, _ConvNd)), "GPTAQ supports only linear and convolutional layers."
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = get_number_of_rows_and_cols(layer)
        # Quantization properties
        self.quantizer = quantizer
        self.quantization_order = QuantizationOrder(quantization_order)
        self.block_size = block_size
        self.rel_damp = rel_damp
        self.alpha = alpha
        self.rtm_lambda = rtm_lambda
        # Backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init matrices
        self.H = None
        self.dXXT = None
        self.num_samples = 0
        # Whether to apply real quantization
        self.export_quantized_model = export_quantized_model

    @torch.no_grad()
    def update(self, input: torch.Tensor, fp_input: torch.Tensor = None) -> None:
        """
        Update the estimate of Hessian matrix and dXXT from a batch of data.
        """
        batch_size = input.shape[0]
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        if self.dXXT is None and fp_input is not None:
            self.dXXT = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)

        if isinstance(self.layer, nn.Linear):
            input = input.reshape(-1, input.shape[-1])
            if fp_input is not None:
                fp_input = fp_input.reshape(-1, fp_input.shape[-1])
        else:
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            input = unfold(input).transpose(1, 2).flatten(0, 1)
            if fp_input is not None:
                fp_input = unfold(fp_input).transpose(1, 2).flatten(0, 1)

        input = input.float()
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha_scale = 2.0 / (self.num_samples + batch_size)
        
        self.H.mul_(beta)
        # We use accumulate_hessian as a helper for H += input.T @ input with scaling
        # However, accumulate_hessian in original code uses triton or bmm.
        # Let's just use torch.addmm for simplicity and consistency with 2gptaq scaling
        # Wait, NVFP's accumulate_hessian has specific scaling.
        # Let's match NVFP's accumulate_hessian style but add dXXT.
        
        tmp_input = input * math.sqrt(alpha_scale)
        self.H.addmm_(tmp_input.t(), tmp_input)

        if fp_input is not None and self.dXXT is not None:
            dX = fp_input.float() * math.sqrt(alpha_scale) - tmp_input
            self.dXXT.mul_(beta)
            self.dXXT.addmm_(dX.t(), tmp_input)

        self.num_samples += batch_size

    def reset(self) -> None:
        self.W = self.layer.weight
        self.H = None
        self.dXXT = None
        self.num_samples = 0
        clear_device_cache()

    def free(self) -> None:
        self.H = None
        self.dXXT = None
        torch.cuda.empty_cache()
        clear_device_cache(garbage_collection=False)

    @torch.no_grad()
    def step(self) -> torch.Tensor | Optional[torch.Tensor] | torch.Tensor:
        d_col, block_size, device, dtype = self.d_col, self.block_size, self.W_device, self.W_dtype
        quantizer_group_size = self.quantizer.group_size
        group_size = quantizer_group_size or d_col
        num_groups = d_col // group_size

        qweight = None
        if self.export_quantized_model:
            qweight = torch.empty(self.W.shape, device=device, dtype=dtype)
        
        # Prepare weight
        self.W = self.layer.weight.clone().float()
        if isinstance(self.layer, _ConvNd):
            self.W = self.W.flatten(1, -1)
            
        scales, zeros = self.quantizer.get_quantization_params(self.W) 
        self.quantizer.group_size = None
        
        if self.quantization_order == QuantizationOrder.ACTIVATION:
            perm = torch.argsort(self.H.diag(), descending=True)
            group_idx = torch.arange(num_groups, device=device).repeat_interleave(group_size)[perm]
            permuted_group_idx = group_idx
        else:
            perm = torch.arange(d_col, device=device)
            permuted_group_idx = torch.arange(d_col, device=device) // group_size
        perm_inv = torch.argsort(perm)
        
        H = self.H[perm][:, perm]
        if self.dXXT is not None:
            dXXT = self.dXXT[perm][:, perm]
        else:
            dXXT = None
            
        w = self.W[:, perm]
        
        # Hessian inverse
        zero_cols = torch.nonzero(self.W.eq(0).all(dim=0))
        H_reg = H.clone()
        H_reg[zero_cols, :] = 0
        H_reg[:, zero_cols] = 0
        H_reg[zero_cols, zero_cols] = 1
        damp = self.rel_damp * torch.diag(H_reg).mean()
        H_reg[range(d_col), range(d_col)] += damp
        
        try:
            H_inv = inv_sym(H_reg)
            H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
        except:
            H_inv_cho = torch.eye(d_col, device=H.device, dtype=torch.float32)

        # GPTAQ P matrix
        if dXXT is not None:
            if self.alpha == 'auto':
                signal_ratio = torch.norm(dXXT, p='fro') / (torch.norm(H, p='fro') + 1e-8)
                alpha_val = min(signal_ratio.item(), 1.0)
            else:
                alpha_val = float(self.alpha)
            P = alpha_val * ((dXXT @ H_inv_cho.T).triu_(diagonal=1)) @ H_inv_cho
        else:
            P = None

        if self.rtm_lambda is not None:
            group_sums = torch.zeros((self.d_row, num_groups), device=device, dtype=dtype)
        else:
            group_sums = None

        for c1 in range(0, d_col, block_size):
            c2 = min(c1 + block_size, d_col)
            ncols = c2 - c1
            w_blk = w[:, c1:c2].clone()  
            errs = torch.zeros_like(w_blk)
            H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]
            if P is not None:
                P_blk = P[c1:c2, c1:c2]

            for i in range(ncols):
                w_ci = w_blk[:, i]
                d = H_inv_cho_blk[i, i]
                g_idx = permuted_group_idx[c1 + i]
                
                w_ci_shifted = w_ci
                if group_sums is not None:
                    current_err = group_sums[:, g_idx]
                    d_sq = d ** 2
                    shift = (self.rtm_lambda * d_sq) / (1 + self.rtm_lambda * d_sq) * current_err
                    w_ci_shifted = w_ci - shift
                
                if self.export_quantized_model:
                    q = self.quantizer.quantize(w_ci_shifted, scales[:, g_idx], zeros[:, g_idx])
                    w_q = self.quantizer.dequantize(q, scales[:, g_idx], zeros[:, g_idx])
                    qweight[:, c1 + i] = q
                else:
                    w_q = self.quantizer(w_ci_shifted, scales[:, g_idx], zeros[:, g_idx])
                w[:, c1 + i] = w_q
                
                if group_sums is not None:
                    group_sums[:, g_idx] += (w_q - w_ci)
                
                err = (w_ci - w_q) / d
                # Weight update with P correction
                if P is not None:
                    w_blk[:, i:] -= err.unsqueeze(1).matmul(H_inv_cho_blk[i, i:].unsqueeze(0)) - w_ci.unsqueeze(1).matmul(P_blk[i, i:].unsqueeze(0))
                else:
                    w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                errs[:, i] = err
                
            if P is not None:
                w[:, c2:] -= errs.matmul(H_inv_cho[c1:c2, c2:]) - w_blk.matmul(P[c1:c2, c2:])
            else:
                w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

        w = w[:, perm_inv].contiguous()
        if qweight is not None:
            qweight = qweight[:, perm_inv].contiguous()
        self.quantizer.group_size = quantizer_group_size
        return w.to(dtype), qweight, scales

    def quantize(self) -> torch.Tensor | Optional[torch.Tensor] | torch.Tensor:
        return self.step()


def gptaq_quantization(
    model: AutoModelForCausalLM, 
    calibration_data: List[torch.Tensor],
    args: argparse.Namespace, 
    device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    print("GPTAQ quantization...")
    orig_dtype = model.config.torch_dtype if args.dtype == "auto" else args.dtype
    act_offload_device = "cpu" if args.cpu_offload_activations else device
    quantized_state_dict = {}
    non_quantized_state_dict = {}
    transform_kwargs = dict(device=device, group_size=args.hadamard_group_size)
    
    weight_quantizer_kwargs = None
    if args.w_bits < 16:
        weight_quantizer_kwargs = dict(
            bits=args.w_bits, 
            symmetric=True, 
            format=args.format,
            granularity=args.w_granularity,
            observer=args.w_observer, 
            group_size=args.w_group_size,
            scale_precision=args.scale_precision
        )
    act_quantizer_kwargs = None
    if args.a_bits < 16:
        act_quantizer_kwargs = dict(
            bits=args.a_bits,
            symmetric=True, 
            format=args.format,
            granularity=args.a_granularity,
            observer=args.a_observer, 
            group_size=args.a_group_size,
            scale_precision=args.scale_precision
        )

    blocks = model.model.layers
    blocks[0] = InputCollector(blocks[0], cpu_offload=args.cpu_offload_activations)
    if args.cpu_offload_modules:
        model.get_input_embeddings().to(device)
        blocks[0] = blocks[0].to(device)

    for sample in calibration_data:
        try:
            with torch.no_grad():
                model(sample.to(device=device))
        except ForwardInterrupt:
            pass
        
    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module

    if args.cpu_offload_modules:
        model.get_input_embeddings().cpu()

    for block_idx, block in enumerate(blocks):
        print(f"Processing block {block_idx}...")
        if args.cpu_offload_modules:
            block.to(device)

        qkv_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        o_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        gate_up_in_transform = build_transform(args.transform_class, size=model.config.hidden_size, **transform_kwargs)
        down_in_transform = build_transform(args.transform_class, size=model.config.intermediate_size, **transform_kwargs)     

        quantized_attn = get_attention_layer(model.config)(
            model.config, layer_idx=block_idx, act_quantizer_kwargs=act_quantizer_kwargs,
            qkv_in_transform=qkv_in_transform, o_in_transform=o_in_transform
        )
        quantized_mlp = get_mlp_layer(model.config)(
            model.config, act_quantizer_kwargs=act_quantizer_kwargs,
            gate_up_in_transform=gate_up_in_transform, down_in_transform=down_in_transform
        )
        quantized_attn.load_state_dict(block.self_attn.state_dict(), strict=False)
        quantized_mlp.load_state_dict(block.mlp.state_dict(), strict=False)
        block.self_attn = quantized_attn
        block.mlp = quantized_mlp
        block = block.to(device=device, dtype=orig_dtype)
        block.requires_grad_(False)

        qkv_in_transform.remove_parametrizations()
        o_in_transform.remove_parametrizations()
        gate_up_in_transform.remove_parametrizations()
        down_in_transform.remove_parametrizations() 

        gptaq_handles = {}
        for layer_name, layer in block.named_modules():
            if isinstance(layer, QLinear):
                gptaq_handles[layer_name] = GPTAQ(
                    layer, Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None, 
                    quantization_order=args.quantization_order, rel_damp=args.rel_damp,
                    export_quantized_model=args.export_quantized_model, alpha=args.alpha,
                    rtm_lambda=args.RTM
                )

        # Transform weights before quantization (do this before ANY pass)
        block.self_attn.q_proj.weight.data = qkv_in_transform(block.self_attn.q_proj.weight, inv_t=True)
        block.self_attn.k_proj.weight.data = qkv_in_transform(block.self_attn.k_proj.weight, inv_t=True)
        block.self_attn.v_proj.weight.data = qkv_in_transform(block.self_attn.v_proj.weight, inv_t=True)
        block.self_attn.o_proj.weight.data = o_in_transform(block.self_attn.o_proj.weight, inv_t=True)
        block.mlp.gate_proj.weight.data = gate_up_in_transform(block.mlp.gate_proj.weight, inv_t=True)
        block.mlp.up_proj.weight.data = gate_up_in_transform(block.mlp.up_proj.weight, inv_t=True)
        block.mlp.down_proj.weight.data = down_in_transform(block.mlp.down_proj.weight, inv_t=True)
        
        orig_fp_weights = {}
        quantized_weights = {}
        for layer_name, layer in block.named_modules():
            if isinstance(layer, QLinear):
                orig_fp_weights[layer_name] = layer.weight.data.clone().cpu()
                quantized_weights[layer_name] = layer.weight.data.clone()
                layer._train_mode = False
                # if layer.act_quantizer: layer.act_quantizer._track_global_scale = False

        print("  Sequential Accumulation and Quantization...")
        sequential_groups = [
            [k for k in gptaq_handles.keys() if k in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']],
            [k for k in gptaq_handles.keys() if k == 'self_attn.o_proj'],
            [k for k in gptaq_handles.keys() if k in ['mlp.gate_proj', 'mlp.up_proj']],
            [k for k in gptaq_handles.keys() if k == 'mlp.down_proj']
        ]

        device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

        for group in sequential_groups:
            if not group: continue

            fp_cache = {}
            def hook_factory(name):
                def _hook(_, inp, out):
                    fp_cache[name] = inp[0].detach()
                return _hook

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                # --- A. Full Precision Pass ---
                for name, layer in block.named_modules():
                    if isinstance(layer, QLinear):
                        layer.weight.data = orig_fp_weights[name].to(device)

                hooks = [dict(block.named_modules())[name].register_forward_hook(hook_factory(name)) for name in group]

                with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                    block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                
                for h in hooks: h.remove()
                fp_inputs = {name: fp_cache[name].clone() for name in group} 

                # --- B. Quantized Pass ---
                for name, layer in block.named_modules():
                    if isinstance(layer, QLinear):
                        layer.weight.data = quantized_weights[name]

                hooks = [dict(block.named_modules())[name].register_forward_hook(hook_factory(name)) for name in group]

                with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                    block(*to(inp_args, device=device), **to(inp_kwargs, device=device))

                for h in hooks: h.remove()

                # Accumulate H and dXXT
                for name in group:
                    gptaq_handles[name].update(fp_cache[name], fp_input=fp_inputs[name])

            # Quantize the current group
            for layer_name in group:
                gptaq_handle = gptaq_handles[layer_name]
                dequantized_qweight, qweight, scales = gptaq_handle.quantize()
                
                with torch.no_grad():
                    relative_mse_error = get_relative_mse_error(
                        dequantized_qweight.float(), 
                        orig_fp_weights[layer_name].to(device).float(), 
                        gptaq_handle.H
                    )
                print(f"[{layer_name:16}]: Relative MSE error: {relative_mse_error.item():.2e}")
                
                # PERMANENTLY update the weights
                quantized_weights[layer_name] = dequantized_qweight
                gptaq_handle.layer.weight.data = dequantized_qweight
                
                if args.export_quantized_model:
                    weight_global_scale = gptaq_handle.quantizer.global_scale.to(scales.device)
                    act_global_scale = gptaq_handle.layer.act_quantizer.global_scale
                    transform_matrix = get_transform_matrix(args.transform_class, args.hadamard_group_size, device, orig_dtype).cpu()
                    if args.export_quantized_model == "realquant":
                        quantized_state_dict[f"model.layers.{block_idx}.{layer_name}"] = {
                            "qweight": pack_fp4_to_uint8(qweight).cpu(),
                            "scales": cast_scales_to_eXmY(scales * weight_global_scale, args.scale_precision).cpu(),
                            "forward_hadamard_matrix": transform_matrix, "backward_hadamard_matrix": transform_matrix.clone(),
                            "weight_global_scale": weight_global_scale.clone(), "act_global_scale": act_global_scale.clone()
                        }
                    else:
                        quantized_state_dict[f"model.layers.{block_idx}.{layer_name}"] = {
                            "dqweight": dequantized_qweight.cpu(),
                            "forward_hadamard_matrix": transform_matrix, "backward_hadamard_matrix": transform_matrix.clone(),
                            "weight_global_scale": weight_global_scale.clone(), "act_global_scale": act_global_scale.clone()
                        }
                
                gptaq_handle.free()
        # Freeze global scale tracking after quantization is complete
        for layer_name, layer in block.named_modules():
            if isinstance(layer, QLinear):
                if layer.act_quantizer: layer.act_quantizer._track_global_scale = False        

        # Update activations for next block
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=args.amp):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            out = maybe_first_element(out).to(act_offload_device)
            if len(inp_args) > 0: inp_args[0].data = out
            elif "hidden_states" in inp_kwargs: inp_kwargs["hidden_states"] = out
            else: raise ValueError("Unsupported block input format.")

        if args.cpu_offload_modules: block = block.cpu()
        del gptaq_handles
        clear_device_cache(garbage_collection=True)

    clear_device_cache(garbage_collection=True)
    return quantized_state_dict, non_quantized_state_dict
