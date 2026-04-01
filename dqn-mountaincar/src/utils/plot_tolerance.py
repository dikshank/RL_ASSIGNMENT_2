# plot_tolerance.py (FIXED)

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
from scipy.stats import norm


def align_timesteps(runs, num_points=500):
    max_t = min(r["timestep"].max() for r in runs)
    common_ts = np.linspace(0, max_t, num_points)

    aligned = []
    for r in runs:
        r = r.sort_values("timestep")
        interp = np.interp(common_ts, r["timestep"], r["return"])
        aligned.append(interp)

    return common_ts, np.array(aligned)


def load(path):
    files = glob.glob(f"{path}/seed_*.csv")
    runs = [pd.read_csv(f) for f in files]

    print(f"[INFO] Loaded {len(runs)} runs from {path}")
    return align_timesteps(runs)


# Approximate tolerance interval (normal assumption)
def tolerance_interval(data, alpha=0.05, beta=0.9):
    n = data.shape[0]

    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)

    # tolerance factor (approx)
    z = norm.ppf((1 + beta) / 2)  # ~1.645 for 90%
    k = z * np.sqrt(1 + 1/n)

    lower = mean - k * std
    upper = mean + k * std

    return mean, lower, upper


def plot():
    rhos = [1, 2, 4, 8]

    plt.figure(figsize=(10,6))

    for r in rhos:
        print(f"[INFO] Processing rho={r}")

        x, data = load(f"logs/raw/rho_{r}")
        mean, low, high = tolerance_interval(data)

        plt.plot(x, mean, label=f"ρ={r}")
        plt.fill_between(x, low, high, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.title("Tolerance Intervals (α=0.05, β=0.9)")
    plt.legend()
    plt.grid()

    os.makedirs("results/plots", exist_ok=True)
    save_path = "results/plots/dqn_rho_tolerance_intervals.png"

    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved → {save_path}")

    plt.show()


if __name__ == "__main__":
    plot()

# import numpy as np
# import pandas as pd
# import glob
# import matplotlib.pyplot as plt
# import os

# def load(path):
#     files = glob.glob(f"{path}/seed_*.csv")
#     runs = [pd.read_csv(f) for f in files]

#     min_len = min(len(r) for r in runs)

#     returns = np.array([r["return"].values[:min_len] for r in runs])
#     timesteps = runs[0]["timestep"].values[:min_len]

#     return timesteps, returns
    
# def plot():
#     rhos = [1,2,4,8]

#     plt.figure()

#     for r in rhos:
#         x, data = load(f"logs/raw/rho_{r}")
#         mean = data.mean(axis=0)
#         low = np.percentile(data, 5, axis=0)
#         high = np.percentile(data, 95, axis=0)

#         x = np.arange(len(mean))
#         plt.plot(x, mean, label=f"ρ={r}")
#         plt.fill_between(x, low, high, alpha=0.2)

#     plt.legend()
#     plt.title("Tolerance Intervals")
#     plt.savefig("results/plots/tolerance.png", dpi=300)
#     plt.show()

# if __name__ == "__main__":
#     plot()