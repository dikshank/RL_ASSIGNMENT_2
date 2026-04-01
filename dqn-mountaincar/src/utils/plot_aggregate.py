# plot_rho_aggregate.py 

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

    return np.array(aucs)


def stats(a):
    mean = np.mean(a)
    ci = 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))
    return mean, ci


def plot():
    rhos = [1, 2, 4, 8]
    means, cis = [], []

    for r in rhos:
        print(f"[INFO] Processing rho={r}")

        a = load_auc(f"logs/raw/rho_{r}")
        m, c = stats(a)

        means.append(m)
        cis.append(c)

    x = np.arange(len(rhos))

    plt.figure()
    plt.errorbar(x, means, yerr=cis, fmt='o-', capsize=5)

    plt.xticks(x, [f"ρ={r}" for r in rhos])
    plt.ylabel("AUC (Time-normalized)")
    plt.title("Aggregate Performance vs Replay Factor")
    plt.grid()

    os.makedirs("results/plots", exist_ok=True)
    save_path = "results/plots/dqn_rho_aggregate.png"

    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved → {save_path}")

    plt.show()


if __name__ == "__main__":
    plot()
    
    
# # Q4(a) — Aggregate Comparison (Figure 3b)

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
#     means, cis = [], []

#     for r in rhos:
#         a = auc(f"logs/raw/rho_{r}")
#         m,c = stats(a)
#         means.append(m)
#         cis.append(c)

#     x = np.arange(len(rhos))

#     plt.figure()
#     plt.errorbar(x, means, yerr=cis, fmt='o-')
#     plt.xticks(x, [f"ρ={r}" for r in rhos])
#     plt.title("Aggregate Performance (AUC)")
#     plt.savefig("results/plots/rho_aggregate.png", dpi=300)
#     plt.show()

# if __name__ == "__main__":
#     plot()