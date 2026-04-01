import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob


def load_runs(log_dir):
    files = glob.glob(f"{log_dir}/seed_*.csv")
    runs = [pd.read_csv(f) for f in files]
    return runs


def align_runs(runs):
    min_len = min(len(r) for r in runs)
    returns = np.array([r["return"].values[:min_len] for r in runs])
    return returns


def compute_ci(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    n = data.shape[0]
    ci = 1.96 * std / np.sqrt(n)
    return mean, ci


def plot_multi(log_dir, save_path=None):
    runs = load_runs(log_dir)
    data = align_runs(runs)

    mean, ci = compute_ci(data)
    x = np.arange(len(mean))

    plt.figure(figsize=(10, 5))
    plt.plot(x, mean, label="Mean Return")
    plt.fill_between(x, mean - ci, mean + ci, alpha=0.3, label="95% CI")

    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("DQN on MountainCar (Mean + 95% CI)")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)

    plt.show()


if __name__ == "__main__":
    plot_multi("logs/raw/baseline", "results/plots/baseline_ci.png")