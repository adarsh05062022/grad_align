import numpy as np
import torch
import pandas as pd


class GradientStatsLogger:
    def __init__(self):
        self.stats = {
            "cos_rf": [],
            "cos_r_tilde": [],
            "cos_f_tilde": [],
            "ratio": [],
            "norm_r": [],
            "norm_f": [],
            "alpha_r": [],
            "alpha_f": [],
            "step": [],
        }

    def log(self, gr, gf, alpha_r, alpha_f, step):
        """
        gr: retain gradient (flat tensor)
        gf: forget gradient (flat tensor)
        alpha_r, alpha_f: Nash weights
        """

        norm_r = torch.norm(gr)
        norm_f = torch.norm(gf)

        cos_rf = torch.dot(gr, gf) / (norm_r * norm_f + 1e-8)

        g_tilde = alpha_r * gr + alpha_f * gf
        norm_tilde = torch.norm(g_tilde)

        cos_r_tilde = torch.dot(gr, g_tilde) / (norm_r * norm_tilde + 1e-8)
        cos_f_tilde = torch.dot(gf, g_tilde) / (norm_f * norm_tilde + 1e-8)

        ratio = (alpha_r * norm_r.item()) / (alpha_f * norm_f.item() + 1e-8)

        # Store values
        self.stats["cos_rf"].append(cos_rf.item())
        self.stats["cos_r_tilde"].append(cos_r_tilde.item())
        self.stats["cos_f_tilde"].append(cos_f_tilde.item())
        self.stats["ratio"].append(ratio)
        self.stats["norm_r"].append(norm_r.item())
        self.stats["norm_f"].append(norm_f.item())
        self.stats["alpha_r"].append(float(alpha_r))
        self.stats["alpha_f"].append(float(alpha_f))
        self.stats["step"].append(step)

    def save_npy(self, path):
        
        np.save(path, self.stats)
        print(f"Saved gradient stats to {path}")

    def save_csv(self, path):
        df = pd.DataFrame(self.stats)
        df.to_csv(path, index=False)
        print(f"Saved gradient stats to {path}")

    def print_variances(self):
        print("==== Variance Diagnostics ====")
        for key in self.stats:
            if key != "step":
                print(f"{key}: {np.var(self.stats[key])}")
