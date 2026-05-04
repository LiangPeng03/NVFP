import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class PreprocessingStatsCollector:
    def __init__(self, hadamard_group_size=128):
        self.hadamard_group_size = hadamard_group_size
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

        G = self.hadamard_group_size
        out_f, in_f = W.shape
        
        # 1. Metric 1: Mean of Channel Means (Column-wise)
        # Col means: [in_features]
        col_means = W.mean(dim=0)
        # Group them into [in_features // G, G]
        num_groups = in_f // G
        grouped_col_means = col_means[:num_groups * G].view(num_groups, G).mean(dim=1)
        self.stats[module_name][phase]["col_means"].extend(grouped_col_means.tolist())

        # 2. Metric 2: Actual Group Means (Row-wise)
        # For each row, group G elements.
        # W_grouped: [out_features, in_features // G, G]
        W_grouped = W[:, :num_groups * G].view(out_f, num_groups, G)
        row_group_means = W_grouped.mean(dim=2) # [out_features, num_groups]
        self.stats[module_name][phase]["row_means"].extend(row_group_means.flatten().tolist())

        # 3. Metric 3: Mean of Channel Variances
        col_vars = W.var(dim=0)
        grouped_col_vars = col_vars[:num_groups * G].view(num_groups, G).mean(dim=1)
        self.stats[module_name][phase]["col_vars"].extend(grouped_col_vars.tolist())

    def plot_all(self, save_dir="./preprocessing_plots"):
        os.makedirs(save_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
        for mod, phases in self.stats.items():
            print(f"Generating plots for {mod}...")
            # 1. Algorithm Convergence (Metric 1)
            plt.figure(figsize=(10, 6))
            for p in ["Original", "Preprocessed"]:
                sns.kdeplot(phases[p]["col_means"], label=p, fill=True)
            plt.axvline(0, color='red', linestyle='--')
            plt.title(f"{mod} - Distribution of Grouped Channel Means (Target Optimization)")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{mod}_col_means.png"), dpi=300)
            plt.close()

            # 2. Actual Quantization Error / DC Outliers (Metric 2)
            plt.figure(figsize=(10, 6))
            for p in ["Original", "Preprocessed", "Rotated"]:
                sns.kdeplot(phases[p]["row_means"], label=p, fill=True)
            plt.axvline(0, color='red', linestyle='--')
            plt.title(f"{mod} - Distribution of Actual Group Means (DC Outlier Magnitude)")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{mod}_group_means.png"), dpi=300)
            plt.close()

            # 3. Variance Correlation (Scatter)
            # We take Preprocessed stats
            vars_v = np.array(phases["Preprocessed"]["col_vars"])
            # row_means is [out_f * num_groups]. col_vars is [num_groups].
            # We need to broadcast col_vars to match row_means.
            num_groups = len(vars_v)
            out_f = len(phases["Preprocessed"]["row_means"]) // num_groups
            
            # Subsample for plotting to avoid freezing
            indices = np.random.choice(len(phases["Preprocessed"]["row_means"]), min(10000, len(phases["Preprocessed"]["row_means"])), replace=False)
            sampled_row_means = np.array(phases["Preprocessed"]["row_means"])[indices]
            # Map index to group_index
            sampled_vars = vars_v[indices % num_groups]
            
            plt.figure(figsize=(10, 6))
            sns.regplot(x=sampled_vars, y=np.abs(sampled_row_means), scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
            plt.xlabel("Average Channel Variance in Group")
            plt.ylabel("Absolute Row-Group Mean (DC Residue)")
            plt.title(f"{mod} - Correlation: Why Preprocessing Fails (Variance vs DC Residue)")
            plt.savefig(os.path.join(save_dir, f"{mod}_variance_correlation.png"), dpi=300)
            plt.close()
            
            # Print Console Summary
            raw_err = np.abs(phases["Original"]["row_means"]).mean()
            prep_err = np.abs(phases["Preprocessed"]["row_means"]).mean()
            rot_err = np.abs(phases["Rotated"]["row_means"]).mean()
            avg_var = np.mean(phases["Preprocessed"]["col_vars"])
            imp = (raw_err - prep_err) / (raw_err + 1e-8) * 100
            print(f"| {mod:10} | Raw_Err: {raw_err:.4f} | Prep_Err: {prep_err:.4f} | Rot_Err: {rot_err:.4f} | Imp: {imp:6.1f}% | Var: {avg_var:.4f} |")
