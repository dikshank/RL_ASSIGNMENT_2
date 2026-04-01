# analyze_truncation.py

import numpy as np
import pandas as pd
import glob
import itertools
from scipy import stats


def compute_auc(df):
    timesteps = df["timestep"].values
    returns = df["return"].values

    # Normalize time to [0,1]
    t_norm = timesteps / timesteps.max()

    # Trapezoidal integration
    return np.trapz(returns, t_norm)


def load_metrics(path):
    files = glob.glob(f"{path}/seed_*.csv")

    aucs = []
    finals = []
    success_rates = []

    for f in files:
        df = pd.read_csv(f)
        df = df.sort_values("timestep")

        # --- AUC (correct) ---
        aucs.append(compute_auc(df))

        # --- Final performance ---
        finals.append(np.mean(df["return"].values[-50:]))

        # --- Success rate ---
        if "success" in df.columns:
            success_rates.append(np.mean(df["success"].values))

    return (
        np.array(aucs),
        np.array(finals),
        np.array(success_rates)
    )


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x) + (ny-1)*np.var(y)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def summarize(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values))
    }


def analyze():
    truncs = [200, 1000, 2000]

    auc_data = {}
    final_data = {}
    success_data = {}

    print("\n========== TRUNCATION ANALYSIS (FIXED) ==========")

    for t in truncs:
        path = f"logs/raw/trunc_{t}"
        print(f"\n[INFO] Loading truncation = {t}")

        auc, final, success = load_metrics(path)

        auc_data[t] = auc
        final_data[t] = final
        success_data[t] = success

        print(f"AUC: {summarize(auc)}")
        print(f"Final: {summarize(final)}")
        print(f"Success: {summarize(success)}")

    print("\n========== PAIRWISE TESTS ==========")

    for (t1, t2) in itertools.combinations(truncs, 2):
        print(f"\n--- {t1} vs {t2} ---")

        # AUC
        t_stat, p_val = stats.ttest_ind(auc_data[t1], auc_data[t2])
        d = cohens_d(auc_data[t1], auc_data[t2])
        print(f"[AUC] t={t_stat:.4f}, p={p_val:.6f}, d={d:.4f}")

        # Final
        t_stat_f, p_val_f = stats.ttest_ind(final_data[t1], final_data[t2])
        d_f = cohens_d(final_data[t1], final_data[t2])
        print(f"[Final] t={t_stat_f:.4f}, p={p_val_f:.6f}, d={d_f:.4f}")

        # Success
        t_stat_s, p_val_s = stats.ttest_ind(success_data[t1], success_data[t2])
        d_s = cohens_d(success_data[t1], success_data[t2])
        print(f"[Success] t={t_stat_s:.4f}, p={p_val_s:.6f}, d={d_s:.4f}")


if __name__ == "__main__":
    analyze()

# import numpy as np
# import pandas as pd
# import glob
# import itertools
# import os
# from scipy import stats


# def load_aucs(path):
#     files = glob.glob(f"{path}/seed_*.csv")
#     aucs = []

#     for f in files:
#         df = pd.read_csv(f)
#         returns = df["return"].values
#         auc = np.mean(returns)
#         aucs.append(auc)

#     return np.array(aucs)


# def load_final_returns(path, last_k=50):
#     files = glob.glob(f"{path}/seed_*.csv")
#     finals = []

#     for f in files:
#         df = pd.read_csv(f)
#         returns = df["return"].values
#         finals.append(np.mean(returns[-last_k:]))

#     return np.array(finals)


# # ✅ NEW: success loader
# def load_success_rates(path):
#     files = glob.glob(f"{path}/seed_*.csv")
#     rates = []

#     for f in files:
#         df = pd.read_csv(f)

#         if "success" not in df.columns:
#             raise ValueError(f"'success' column not found in {f}")

#         success = df["success"].values
#         rates.append(np.mean(success))  # per-seed success rate

#     return np.array(rates)


# def cohens_d(x, y):
#     nx, ny = len(x), len(y)
#     dof = nx + ny - 2
#     pooled_std = np.sqrt(((nx-1)*np.var(x) + (ny-1)*np.var(y)) / dof)
#     return (np.mean(x) - np.mean(y)) / pooled_std


# def summarize(values):
#     return {
#         "mean": float(np.mean(values)),
#         "std": float(np.std(values)),
#         "min": float(np.min(values)),
#         "max": float(np.max(values))
#     }


# def analyze():
#     truncs = [200, 1000, 2000]

#     auc_data = {}
#     final_data = {}
#     success_data = {}

#     print("\n=== Loading Data ===")

#     for t in truncs:
#         path = f"logs/raw/trunc_{t}"

#         auc_data[t] = load_aucs(path)
#         final_data[t] = load_final_returns(path)
#         success_data[t] = load_success_rates(path)

#         print(f"\nTruncation {t}")
#         print("AUC:", summarize(auc_data[t]))
#         print("Final:", summarize(final_data[t]))
#         print("Success rate:", summarize(success_data[t]))

#     print("\n=== Pairwise Statistical Tests ===")

#     for (t1, t2) in itertools.combinations(truncs, 2):

#         # ---------- AUC ----------
#         t_stat, p_val = stats.ttest_ind(auc_data[t1], auc_data[t2])
#         d = cohens_d(auc_data[t1], auc_data[t2])

#         print(f"\n[AUC] {t1} vs {t2}")
#         print(f"t-stat: {t_stat:.4f}, p-value: {p_val:.6f}, Cohen's d: {d:.4f}")

#         # ---------- Final ----------
#         t_stat_f, p_val_f = stats.ttest_ind(final_data[t1], final_data[t2])
#         d_f = cohens_d(final_data[t1], final_data[t2])

#         print(f"[Final] {t1} vs {t2}")
#         print(f"t-stat: {t_stat_f:.4f}, p-value: {p_val_f:.6f}, Cohen's d: {d_f:.4f}")

#         # ---------- Success ----------
#         t_stat_s, p_val_s = stats.ttest_ind(success_data[t1], success_data[t2])
#         d_s = cohens_d(success_data[t1], success_data[t2])

#         print(f"[Success] {t1} vs {t2}")
#         print(f"t-stat: {t_stat_s:.4f}, p-value: {p_val_s:.6f}, Cohen's d: {d_s:.4f}")


# if __name__ == "__main__":
#     analyze()