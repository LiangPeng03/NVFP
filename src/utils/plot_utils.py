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
            self.stats[module_name] = {p: {"col_means": [], "row_means": [], "col_vars": []} for p in ["Original", "Preprocessed", "Rotated"]}
        
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
            raw_err = np.abs(phases["Original"]["row_means"]).mean()
            prep_err = np.abs(phases["Preprocessed"]["row_means"]).mean()
            rot_err = np.abs(phases["Rotated"]["row_means"]).mean()
            
            imp_prep = (raw_err - prep_err) / (raw_err + 1e-8) * 100
            imp_rot = (raw_err - rot_err) / (raw_err + 1e-8) * 100
            print(f"| {mod:10} | Raw_Err: {raw_err:.4f} | Prep_Err: {prep_err:.4f} (Imp: {imp_prep:5.1f}%) | Rot_Err: {rot_err:.4f} (Imp: {imp_rot:5.1f}%) |")
