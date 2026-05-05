import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class PreprocessingStatsCollector:
    def __init__(self, w_group_size=16):
        self.w_group_size = w_group_size
        # Structure: stats[module_name][phase][metric_name] = List of values
        self.stats = {}

    @torch.no_grad()
    def collect(self, module_name: str, phase: str, W: torch.Tensor, X_mean: torch.Tensor = None):
        """
        W: [out_features, in_features]
        X_mean: [in_features]
        """
        if module_name not in self.stats:
            self.stats[module_name] = {p: {"row_means": [], "max_abs": []} for p in ["Original", "Preprocessed", "Rotated"]}
        
        W = W.float().cpu()
        if X_mean is not None:
            X_mean = X_mean.float().cpu()

        G = self.w_group_size
        out_f, in_f = W.shape
        
        num_groups = in_f // G
        
        # We only care about the actual group means (Row-wise means of each 16/32 element group)
        # W_grouped: [out_features, num_groups, G]
        W_grouped = W[:, :num_groups * G].view(out_f, num_groups, G)
        row_group_means = W_grouped.mean(dim=2) # [out_features, num_groups]
        self.stats[module_name][phase]["row_means"].extend(row_group_means.flatten().tolist())
        
        # 2. Max absolute value (determines Quantization Scale!)
        max_abs_vals = W_grouped.abs().max(dim=2).values
        self.stats[module_name][phase]["max_abs"].extend(max_abs_vals.flatten().tolist())

    def plot_all(self, save_dir="./preprocessing_plots"):
        os.makedirs(save_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
        for mod, phases in self.stats.items():
            print(f"Generating group means plot for {mod}...")
            
            # Actual Quantization Error / DC Outliers (Metric 2)
            plt.figure(figsize=(10, 6))
            for p in ["Original", "Preprocessed", "Rotated"]:
                sns.kdeplot(phases[p]["row_means"], label=p, fill=True)
            plt.axvline(0, color='red', linestyle='--')
            plt.title(f"{mod} - Distribution of Actual Group Means (DC Outlier Magnitude)")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{mod}_group_means.png"), dpi=300)
            plt.close()
            
            # Print Console Summary
            raw_dc = np.abs(phases["Original"]["row_means"]).mean() if phases["Original"]["row_means"] else 0
            prep_dc = np.abs(phases["Preprocessed"]["row_means"]).mean() if phases["Preprocessed"]["row_means"] else 0
            
            raw_max = np.mean(phases["Original"]["max_abs"]) if phases["Original"]["max_abs"] else 0
            rot_max = np.mean(phases["Rotated"]["max_abs"]) if phases["Rotated"]["max_abs"] else 0
            
            imp_dc = (raw_dc - prep_dc) / (raw_dc + 1e-8) * 100
            imp_scale = (raw_max - rot_max) / (raw_max + 1e-8) * 100
            
            print(f"| {mod:10} | Raw DC: {raw_dc:.4f} -> Prep DC: {prep_dc:.4f} (Imp: {imp_dc:5.1f}%) | Raw Max: {raw_max:.4f} -> Rot Max: {rot_max:.4f} (Scale Imp: {imp_scale:5.1f}%) |")
