# plot_truncation.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os


def load_runs(path):
    files = glob.glob(f"{path}/seed_*.csv")
    print(f"[INFO] Loading {len(files)} runs from {path}")

    runs = []
    for f in files:
        df = pd.read_csv(f)
        df = df.sort_values("timestep")
        runs.append(df)

    return runs


def align_timesteps(runs, num_points=500):
    max_t = min(r["timestep"].max() for r in runs)
    print(f"[INFO] Common max timestep: {max_t}")

    common_ts = np.linspace(0, max_t, num_points)

    aligned = []
    for i, r in enumerate(runs):
        interp_returns = np.interp(common_ts, r["timestep"], r["return"])
        aligned.append(interp_returns)

    print(f"[INFO] Interpolated {len(runs)} runs over {num_points} points")
    return common_ts, np.array(aligned)


def compute_ci(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)
    ci = 1.96 * std / np.sqrt(data.shape[0])

    print(f"[INFO] Computed mean + 95% CI over {data.shape[0]} runs")
    return mean, ci


def plot():
    truncs = [200, 1000, 2000]

    plt.figure(figsize=(10, 6))

    for t in truncs:
        print(f"\n[INFO] Processing truncation = {t}")

        runs = load_runs(f"logs/raw/trunc_{t}")

        if len(runs) == 0:
            print(f"[WARNING] No runs found for trunc_{t}")
            continue

        timesteps, data = align_timesteps(runs)
        mean, ci = compute_ci(data)

        plt.plot(timesteps, mean, label=f"Truncation = {t}")
        plt.fill_between(timesteps, mean - ci, mean + ci, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.title("DQN on MountainCar-v0: Effect of Episode Truncation")
    plt.legend()
    plt.grid()

    os.makedirs("results/plots", exist_ok=True)
    save_path = "results/plots/dqn_truncation_comparison.png"

    plt.savefig(save_path, dpi=300)
    print(f"\n[INFO] Saved plot → {save_path}")

    plt.show()


if __name__ == "__main__":
    plot()


# # Q3.3(a) — Truncation Comparison
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import glob
# import os

# def load_runs(path):
#     files = glob.glob(f"{path}/seed_*.csv")
#     runs = [pd.read_csv(f) for f in files]
#     return runs

# def align_timesteps(runs):
#     min_len = min(len(r) for r in runs)

#     returns = np.array([r["return"].values[:min_len] for r in runs])
#     timesteps = runs[0]["timestep"].values[:min_len]

#     return timesteps, returns

# def compute_ci(data):
#     mean = data.mean(axis=0)
#     std = data.std(axis=0)
#     ci = 1.96 * std / np.sqrt(data.shape[0])
#     return mean, ci

# def plot():
#     truncs = [200, 1000, 2000]

#     plt.figure(figsize=(10,6))

#     for t in truncs:
#         runs = load_runs(f"logs/raw/trunc_{t}")
#         timesteps, data = align_timesteps(runs)

#         mean, ci = compute_ci(data)

#         plt.plot(timesteps, mean, label=f"{t}")
#         plt.fill_between(timesteps, mean-ci, mean+ci, alpha=0.2)

#     plt.xlabel("Timesteps")
#     plt.ylabel("Return")
#     plt.title("Truncation Comparison (Aligned by Timesteps)")
#     plt.legend()
#     plt.grid()

#     os.makedirs("results/plots", exist_ok=True)
#     plt.savefig("results/plots/truncation.png", dpi=300)

#     plt.show()
#     print("saving to 'results/plots/truncation.png' ")


# if __name__ == "__main__":
#     plot()