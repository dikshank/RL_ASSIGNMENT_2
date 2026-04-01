# plot_sensitivity.py

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import re


# ----------- CORRECT AUC -----------
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


def collect(prefix):
    base = "logs/raw"
    res = {}

    for folder in os.listdir(base):
        if folder.startswith(prefix):
            path = f"{base}/{folder}"
            a = load_auc(path)
            res[folder] = stats(a)

    return res


def extract_bs(k):
    return int(re.search(r"bs(\d+)", k).group(1))


def extract_tu(k):
    return int(re.search(r"tu(\d+)", k).group(1))


# ----------- PLOT BATCH -----------
def plot_batch():
    print("\n[INFO] Plotting batch size sensitivity...")
    res = collect("sens_bs")

    rho1, rho4 = {}, {}

    for k, v in res.items():
        if "rho1" in k:
            rho1[extract_bs(k)] = v
        elif "rho4" in k:
            rho4[extract_bs(k)] = v

    plt.figure()

    for data, label in [(rho1, "ρ=1"), (rho4, "ρ=4")]:
        xs = sorted(data)
        ys = [data[x][0] for x in xs]
        es = [data[x][1] for x in xs]

        plt.errorbar(xs, ys, yerr=es, marker='o', capsize=5, label=label)

    plt.xlabel("Batch Size")
    plt.ylabel("Aggregate Performance (AUC)")
    plt.title("Batch Size Sensitivity (DQN)")
    plt.legend()
    plt.grid()

    os.makedirs("results/plots", exist_ok=True)
    save_path = "results/plots/dqn_sensitivity_batch.png"

    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved → {save_path}")
    plt.show()


# ----------- PLOT TARGET -----------
def plot_target():
    print("\n[INFO] Plotting target update sensitivity...")
    res = collect("sens_tu")

    rho1, rho4 = {}, {}

    for k, v in res.items():
        if "rho1" in k:
            rho1[extract_tu(k)] = v
        elif "rho4" in k:
            rho4[extract_tu(k)] = v

    plt.figure()

    for data, label in [(rho1, "ρ=1"), (rho4, "ρ=4")]:
        xs = sorted(data)
        ys = [data[x][0] for x in xs]
        es = [data[x][1] for x in xs]

        plt.errorbar(xs, ys, yerr=es, marker='o', capsize=5, label=label)

    plt.xlabel("Target Update Frequency")
    plt.ylabel("Aggregate Performance (AUC)")
    plt.title("Target Network Update Sensitivity (DQN)")
    plt.legend()
    plt.grid()

    os.makedirs("results/plots", exist_ok=True)
    save_path = "results/plots/dqn_sensitivity_target.png"

    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_batch()
    plot_target()






# # Q4(d) — Sensitivity (Figure 3c)

# import numpy as np
# import pandas as pd
# import glob
# import matplotlib.pyplot as plt
# import os
# import re

# def auc(path):
#     files = glob.glob(f"{path}/seed_*.csv")
#     return np.array([pd.read_csv(f)["return"].mean() for f in files])

# def stats(a):
#     return np.mean(a), 1.96*np.std(a)/np.sqrt(len(a))

# def collect(prefix):
#     base = "logs/raw"
#     res = {}

#     for f in os.listdir(base):
#         if f.startswith(prefix):
#             a = auc(f"{base}/{f}")
#             res[f] = stats(a)

#     return res

# def extract_bs(k): return int(re.search(r"bs(\d+)", k).group(1))
# def extract_tu(k): return int(re.search(r"tu(\d+)", k).group(1))

# def plot_batch():
#     res = collect("sens_bs")
#     rho1, rho4 = {}, {}

#     for k,v in res.items():
#         if "rho1" in k: rho1[extract_bs(k)] = v
#         if "rho4" in k: rho4[extract_bs(k)] = v

#     for data,label in [(rho1,"ρ=1"),(rho4,"ρ=4")]:
#         xs = sorted(data)
#         ys = [data[x][0] for x in xs]
#         es = [data[x][1] for x in xs]
#         plt.errorbar(xs, ys, yerr=es, marker='o', label=label)

#     plt.legend()
#     plt.title("Batch Size Sensitivity")
#     plt.savefig("results/plots/sens_batch.png", dpi=300)
#     plt.show()

# def plot_target():
#     res = collect("sens_tu")
#     rho1, rho4 = {}, {}

#     for k,v in res.items():
#         if "rho1" in k: rho1[extract_tu(k)] = v
#         if "rho4" in k: rho4[extract_tu(k)] = v

#     for data,label in [(rho1,"ρ=1"),(rho4,"ρ=4")]:
#         xs = sorted(data)
#         ys = [data[x][0] for x in xs]
#         es = [data[x][1] for x in xs]
#         plt.errorbar(xs, ys, yerr=es, marker='o', label=label)

#     plt.legend()
#     plt.title("Target Update Sensitivity")
#     plt.savefig("results/plots/sens_target.png", dpi=300)
#     plt.show()

# if __name__ == "__main__":
#     plot_batch()
#     plot_target()