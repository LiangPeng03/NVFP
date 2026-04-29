import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math

from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory (HF or safetensors)")
    parser.add_argument("--layers", type=str, default="0,14,29", help="Comma separated list of layers to analyze")
    parser.add_argument("--group_size", type=int, default=16, help="Group size for analysis")
    parser.add_argument("--hadamard", action="store_true", help="Apply Hadamard transform to weights before analysis")
    parser.add_argument("--hadamard_group_size", type=int, default=128, help="Group size for Hadamard transform")
    parser.add_argument("--save_dir", type=str, default="./dist_plots", help="Directory to save plots")
    return parser.parse_args()

def load_weights(model_path):
    print(f"Loading weights from {model_path}...")
    
    # 1. Try to load as HF model (works for local paths and Hub IDs)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True, device_map="cuda")
        return model.state_dict()
    except Exception as e:
        print(f"  HF from_pretrained failed, trying manual loading: {e}")

    # 2. Manual loading for specific files or local directories
    if os.path.isdir(model_path):
        state_dict = {}
        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                state_dict.update(load_file(os.path.join(model_path, file)))
        if state_dict: return state_dict
        
    if model_path.endswith(".safetensors"):
        return load_file(model_path)
    elif model_path.endswith(".pt") or model_path.endswith(".bin"):
        return torch.load(model_path, map_location="cpu")
    
    raise ValueError(f"Could not load model from {model_path}. Please ensure it is a valid path or HF ID.")

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    state_dict = load_weights(args.model_path)
    layers_to_analyze = [int(x) for x in args.layers.split(",")]
    
    modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Check if weights are stored directly or as qweight + scales
    # NVFP exports might save just 'weight' if not using export_quantized_model, 
    # or 'qweight'/'scales' if using the custom format.
    
    for layer_idx in layers_to_analyze:
        print(f"Analyzing Layer {layer_idx}...")
        
        # Prepare subplots: 4 rows (Mean, Std, Skew, Kurt), len(modules) columns
        fig, axes = plt.subplots(4, len(modules), figsize=(4 * len(modules), 20))
        fig.suptitle(f"Group Distribution (Size={args.group_size}) - Layer {layer_idx}", fontsize=16)
        
        for col, mod in enumerate(modules):
            # Attempt to find the weight tensor
            prefix = f"model.layers.{layer_idx}.self_attn.{mod}"
            if mod in ["gate_proj", "up_proj", "down_proj"]:
                prefix = f"model.layers.{layer_idx}.mlp.{mod}"
                
            w_tensor = None
            if f"{prefix}.weight" in state_dict:
                w_tensor = state_dict[f"{prefix}.weight"].float()
            elif f"{prefix}.qweight" in state_dict and f"{prefix}.w_scales" in state_dict:
                # Basic dequantization if custom format is used
                # Note: this is a fallback approximation if weights are packed
                print(f"Warning: {mod} is packed, analysis requires dequantization. Skipping for now.")
                continue
                
            if w_tensor is None:
                print(f"  Missing {mod} in layer {layer_idx}")
                continue
                
            # Apply Hadamard if requested
            if args.hadamard:
                if hadamard_transform is None:
                    print("  Error: fast_hadamard_transform not installed. Skipping rotation.")
                else:
                    # Hadamard is applied to the input dimension (dim=-1)
                    # We need to reshape to [..., hadamard_group_size]
                    h_gs = args.hadamard_group_size
                    orig_shape = w_tensor.shape
                    if orig_shape[-1] % h_gs == 0:
                        w_tensor = hadamard_transform(
                            w_tensor.view(-1, h_gs).cuda(), 
                            scale=1.0 / math.sqrt(h_gs)
                        ).view(orig_shape).cpu()
                    else:
                        print(f"  Warning: {mod} in_features ({orig_shape[-1]}) not divisible by {h_gs}. Skipping rotation.")

            # Grouping
            # Shape: [out_features, in_features]
            # Flatten to [num_groups, group_size]
            num_elements = w_tensor.numel()
            if num_elements % args.group_size != 0:
                print(f"  Warning: {mod} elements ({num_elements}) not divisible by group_size {args.group_size}")
                # Trim to divisible
                w_tensor = w_tensor.flatten()[: (num_elements // args.group_size) * args.group_size]
                
            w_grouped = w_tensor.view(-1, args.group_size)
            
            # Calculate Mean, Std, Skewness, and Kurtosis
            group_means = w_grouped.mean(dim=1)
            group_stds = w_grouped.std(dim=1)
            group_centered = w_grouped - group_means.unsqueeze(1)
            # skewness = E[(X - μ)^3] / σ^3
            group_skews = (group_centered ** 3).mean(dim=1) / (group_stds ** 3 + 1e-8)
            # excess kurtosis = E[(X - μ)^4] / σ^4 - 3
            group_kurts = (group_centered ** 4).mean(dim=1) / (group_stds ** 4 + 1e-8) - 3
            
            group_means = group_means.cpu().numpy()
            group_stds = group_stds.cpu().numpy()
            group_skews = group_skews.cpu().numpy()
            group_kurts = group_kurts.cpu().numpy()
            
            # Plot Mean Distribution (Density)
            ax_mean = axes[0, col]
            sns.kdeplot(group_means, ax=ax_mean, fill=True, color="blue")
            ax_mean.set_title(f"{mod}\nGroup Means")
            ax_mean.set_xlabel("Mean")
            ax_mean.axvline(0, color='red', linestyle='--') # Mark 0
            
            # Plot Std Distribution (Density)
            ax_std = axes[1, col]
            sns.kdeplot(group_stds, ax=ax_std, fill=True, color="green")
            ax_std.set_title(f"Group Stds")
            ax_std.set_xlabel("Std")
            
            # Plot Skewness Distribution (Density)
            ax_skew = axes[2, col]
            sns.kdeplot(group_skews, ax=ax_skew, fill=True, color="orange")
            ax_skew.set_title(f"Group Skews")
            ax_skew.set_xlabel("Skewness")
            ax_skew.axvline(0, color='red', linestyle='--') # Mark 0
            
            # Plot Kurtosis Distribution (Density)
            ax_kurt = axes[3, col]
            sns.kdeplot(group_kurts, ax=ax_kurt, fill=True, color="purple")
            ax_kurt.set_title(f"Group Kurts")
            ax_kurt.set_xlabel("Excess Kurtosis")
            ax_kurt.axvline(0, color='red', linestyle='--') # Mark 0
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(args.save_dir, f"layer_{layer_idx}_dist.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
