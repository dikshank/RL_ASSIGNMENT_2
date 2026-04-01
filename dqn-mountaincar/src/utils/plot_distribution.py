# src/utils/plot_distribution.py

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os


def compute_auc(df):
    t = df["timestep"].values
    r = df["return"].values
    t_norm = t / t.max()
    return np.trapz(r, t_norm)


def load_aucs(path):
    files = glob.glob(f"{path}/seed_*.csv")

    aucs = []
    for f in files:
        df = pd.read_csv(f).sort_values("timestep")
        aucs.append(compute_auc(df))

    print(f"[INFO] Loaded {len(aucs)} runs from {path}")
    return np.array(aucs)


def plot_distribution():
    rhos = [1, 2, 4, 8]

    plt.figure(figsize=(10, 6))

    for r in rhos:
        path = f"logs/raw/rho_{r}"
        aucs = load_aucs(path)

        if len(aucs) == 0:
            print(f"[WARNING] No data for rho={r}")
            continue

        # Histogram + KDE-style smoothing via density=True
        plt.hist(
            aucs,
            bins=10,
            density=True,
            alpha=0.4,
            label=f"ρ={r}"
        )

    plt.xlabel("Aggregate Performance (AUC)")
    plt.ylabel("Density")
    plt.title("Distribution of Performance across Replay Factors")
    plt.legend()
    plt.grid()

    os.makedirs("results/plots", exist_ok=True)
    save_path = "results/plots/dqn_rho_distribution.png"

    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved → {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_distribution()