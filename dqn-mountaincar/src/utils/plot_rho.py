# plot_rho_curves.py (FIXED)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os


def load_and_align(path, num_points=500):
    files = glob.glob(f"{path}/seed_*.csv")
    runs = []

    for f in files:
        df = pd.read_csv(f).sort_values("timestep")
        runs.append(df)

    max_t = min(r["timestep"].max() for r in runs)
    common_ts = np.linspace(0, max_t, num_points)

    aligned = []
    for r in runs:
        interp = np.interp(common_ts, r["timestep"], r["return"])
        aligned.append(interp)

    return common_ts, np.array(aligned)


def ci(data):
    m = data.mean(axis=0)
    s = data.std(axis=0, ddof=1)
    return m, 1.96 * s / np.sqrt(data.shape[0])


def plot():
    rhos = [1, 2, 4, 8]

    plt.figure(figsize=(10,6))

    for r in rhos:
        print(f"[INFO] Processing rho={r}")

        x, data = load_and_align(f"logs/raw/rho_{r}")
        m, c = ci(data)

        plt.plot(x, m, label=f"ρ={r}")
        plt.fill_between(x, m-c, m+c, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.title("DQN Replay Factor Comparison (Return vs Timesteps)")
    plt.legend()
    plt.grid()

    os.makedirs("results/plots", exist_ok=True)
    save_path = "results/plots/dqn_rho_learning_curves.png"

    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved → {save_path}")

    plt.show()


if __name__ == "__main__":
    plot()


# # Q4(a) — Replay Factor Learning Curves

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import glob
# import os

# def load(path):
#     files = glob.glob(f"{path}/seed_*.csv")
#     runs = [pd.read_csv(f) for f in files]
#     min_len = min(len(r) for r in runs)
#     data = np.array([r["return"].values[:min_len] for r in runs])
#     return data

# def ci(data):
#     m = data.mean(axis=0)
#     s = data.std(axis=0)
#     return m, 1.96*s/np.sqrt(data.shape[0])

# def plot():
#     rhos = [1,2,4,8]

#     plt.figure()

#     for r in rhos:
#         data = load(f"logs/raw/rho_{r}")
#         m, c = ci(data)
#         x = np.arange(len(m))

#         plt.plot(x, m, label=f"ρ={r}")
#         plt.fill_between(x, m-c, m+c, alpha=0.2)

#     plt.legend()
#     plt.title("Replay Factor Comparison")
#     plt.savefig("results/plots/rho_curves.png", dpi=300)
#     plt.show()

# if __name__ == "__main__":
#     plot()