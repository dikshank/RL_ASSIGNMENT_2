# plot_per_comparison.py (FIXED)

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


def load_auc(path):
    files = glob.glob(f"{path}/seed_*.csv")

    aucs = []
    for f in files:
        df = pd.read_csv(f).sort_values("timestep")
        aucs.append(compute_auc(df))

    print(f"[INFO] Loaded {len(aucs)} runs from {path}")
    return np.array(aucs)


def stats(a):
    mean = np.mean(a)
    ci = 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))
    return mean, ci


def plot():
    rhos = [1, 2, 4, 8]

    uniform_means, uniform_ci = [], []
    per_means, per_ci = [], []

    for r in rhos:
        print(f"[INFO] Processing rho={r}")

        u = load_auc(f"logs/raw/rho_{r}")
        p = load_auc(f"logs/raw/per_rho_{r}")

        m1, c1 = stats(u)
        m2, c2 = stats(p)

        uniform_means.append(m1)
        uniform_ci.append(c1)

        per_means.append(m2)
        per_ci.append(c2)

    x = np.arange(len(rhos))

    plt.figure()

    plt.errorbar(x, uniform_means, yerr=uniform_ci, marker='o', capsize=5, label="Uniform")
    plt.errorbar(x, per_means, yerr=per_ci, marker='s', capsize=5, label="PER")

    plt.xticks(x, [f"ρ={r}" for r in rhos])
    plt.xlabel("Replay Factor (ρ)")
    plt.ylabel("Aggregate Performance (AUC)")
    plt.title("PER vs Uniform Sampling (DQN)")
    plt.legend()
    plt.grid()

    os.makedirs("results/plots", exist_ok=True)
    save_path = "results/plots/dqn_per_vs_uniform.png"

    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved → {save_path}")

    plt.show()


if __name__ == "__main__":
    plot()


# # bonus 5
# # Compare Uniform vs PER (aggregate AUC)
# import numpy as np
# import pandas as pd
# import glob
# import matplotlib.pyplot as plt
# import os

# def auc(path):
#     files = glob.glob(f"{path}/seed_*.csv")
#     return np.array([pd.read_csv(f)["return"].mean() for f in files])

# def stats(a):
#     return np.mean(a), 1.96*np.std(a)/np.sqrt(len(a))

# def plot():
#     rhos = [1,2,4,8]

#     uniform_means, uniform_ci = [], []
#     per_means, per_ci = [], []

#     for r in rhos:
#         u = auc(f"logs/raw/rho_{r}")
#         p = auc(f"logs/raw/per_rho_{r}")

#         m1,c1 = stats(u)
#         m2,c2 = stats(p)

#         uniform_means.append(m1)
#         uniform_ci.append(c1)

#         per_means.append(m2)
#         per_ci.append(c2)

#     x = np.arange(len(rhos))

#     plt.figure()

#     plt.errorbar(x, uniform_means, yerr=uniform_ci, marker='o', label="Uniform")
#     plt.errorbar(x, per_means, yerr=per_ci, marker='s', label="PER")

#     plt.xticks(x, [f"ρ={r}" for r in rhos])
#     plt.xlabel("Replay Factor (ρ)")
#     plt.ylabel("Aggregate Performance (AUC)")
#     plt.title("PER vs Uniform Sampling")
#     plt.legend()

#     os.makedirs("results/plots", exist_ok=True)
#     plt.savefig("results/plots/per_vs_uniform.png", dpi=300)

#     plt.show()

# if __name__ == "__main__":
#     plot()