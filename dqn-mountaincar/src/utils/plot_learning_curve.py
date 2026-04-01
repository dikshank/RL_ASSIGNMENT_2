# Q3.2(a) — Learning Curves (Corrected)

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


def load_runs(path):
    files = glob.glob(f"{path}/seed_*.csv")
    print(f"[INFO] Found {len(files)} run files in {path}")
    
    runs = []
    for f in files:
        df = pd.read_csv(f)
        df = df.sort_values("timestep")
        runs.append(df)
    
    return runs


# ----------- CORRECT: ALIGN BY TIMESTEP -----------
def align_by_timestep(runs, num_points=500):
    max_t = min(r["timestep"].max() for r in runs)
    print(f"[INFO] Common max timestep across runs: {max_t}")

    common_ts = np.linspace(0, max_t, num_points)

    aligned = []
    for i, r in enumerate(runs):
        interp_returns = np.interp(common_ts, r["timestep"], r["return"])
        aligned.append(interp_returns)

    print(f"[INFO] Aligned {len(runs)} runs over {num_points} points (timestep)")
    return np.array(aligned), common_ts


# ----------- OPTIONAL: ALIGN BY EPISODE -----------
def align_by_episode(runs):
    min_len = min(len(r) for r in runs)
    print(f"[INFO] Aligning by episode with min length: {min_len}")

    data = np.array([r["return"].values[:min_len] for r in runs])
    x = np.arange(min_len)
    return data, x


# ----------- CI COMPUTATION -----------
def compute_ci(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)
    ci = 1.96 * std / np.sqrt(data.shape[0])

    print(f"[INFO] Computed mean and 95% CI over {data.shape[0]} runs")
    return mean, ci


# ----------- PLOTTING -----------
def plot_all():
    runs = load_runs("logs/raw/baseline")

    if len(runs) == 0:
        print("[ERROR] No log files found. Exiting.")
        return

    os.makedirs("results/plots", exist_ok=True)

    # ===== TIMESTEP PLOT (PRIMARY — REQUIRED) =====
    print("\n[INFO] Generating TIMESTEP-based learning curve...")
    data_ts, x_ts = align_by_timestep(runs)
    mean_ts, ci_ts = compute_ci(data_ts)

    plt.figure()
    plt.plot(x_ts, mean_ts, label="Mean Return")
    plt.fill_between(x_ts, mean_ts - ci_ts, mean_ts + ci_ts, alpha=0.2, label="95% CI")
    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.title("Vanilla DQN on MountainCar-v0 (Return vs Timesteps)")
    plt.legend()
    plt.grid()

    ts_path = "results/plots/dqn_learning_curve_timesteps.png"
    plt.savefig(ts_path, dpi=300)
    print(f"[INFO] Saved timestep plot → {ts_path}")
    plt.close()


    # ===== EPISODE PLOT (SECONDARY — OPTIONAL BUT USEFUL) =====
    print("\n[INFO] Generating EPISODE-based learning curve...")
    data_ep, x_ep = align_by_episode(runs)
    mean_ep, ci_ep = compute_ci(data_ep)

    plt.figure()
    plt.plot(x_ep, mean_ep, label="Mean Return")
    plt.fill_between(x_ep, mean_ep - ci_ep, mean_ep + ci_ep, alpha=0.2, label="95% CI")
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("Vanilla DQN on MountainCar-v0 (Return vs Episodes)")
    plt.legend()
    plt.grid()

    ep_path = "results/plots/dqn_learning_curve_episodes.png"
    plt.savefig(ep_path, dpi=300)
    print(f"[INFO] Saved episode plot → {ep_path}")
    plt.close()

    print("\n[INFO] Plotting complete.")


if __name__ == "__main__":
    plot_all()