# analyze_rho_stats.py

import numpy as np
import pandas as pd
import glob
import itertools
from scipy import stats


def compute_auc(df):
    t = df["timestep"].values
    r = df["return"].values
    t_norm = t / t.max()
    return np.trapz(r, t_norm)


def load_metrics(path):
    files = glob.glob(f"{path}/seed_*.csv")

    aucs, finals = [], []

    for f in files:
        df = pd.read_csv(f).sort_values("timestep")

        aucs.append(compute_auc(df))
        finals.append(np.mean(df["return"].values[-50:]))

    return np.array(aucs), np.array(finals)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x) + (ny-1)*np.var(y)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def summarize(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1))
    }


def analyze():
    rhos = [1, 2, 4, 8]

    auc_data = {}
    final_data = {}

    print("\n===== RHO ANALYSIS =====")

    for r in rhos:
        path = f"logs/raw/rho_{r}"

        auc, final = load_metrics(path)

        auc_data[r] = auc
        final_data[r] = final

        print(f"\nρ={r}")
        print("AUC:", summarize(auc))
        print("Final:", summarize(final))

    print("\n===== PAIRWISE TESTS =====")

    for r1, r2 in itertools.combinations(rhos, 2):
        print(f"\n--- ρ={r1} vs ρ={r2} ---")

        t_auc, p_auc = stats.ttest_ind(auc_data[r1], auc_data[r2])
        d_auc = cohens_d(auc_data[r1], auc_data[r2])

        t_f, p_f = stats.ttest_ind(final_data[r1], final_data[r2])
        d_f = cohens_d(final_data[r1], final_data[r2])

        print(f"[AUC] t={t_auc:.4f}, p={p_auc:.6f}, d={d_auc:.4f}")
        print(f"[Final] t={t_f:.4f}, p={p_f:.6f}, d={d_f:.4f}")


if __name__ == "__main__":
    analyze()