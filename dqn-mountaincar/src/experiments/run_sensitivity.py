# run_sensitivity.py

import torch
import yaml

from src.env.make_env import make_env
from src.agents.dqn_agent import DQNAgent
from src.replay.replay_buffer import ReplayBuffer
from src.training.train import train
from src.utils.logger import CSVLogger


def run_experiment(config, seed, tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(
        config["env"]["name"],
        config["env"]["max_episode_steps"],
        seed=seed
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, config, device)
    buffer = ReplayBuffer(state_dim, config["train"]["buffer_size"])

    log_path = f"logs/raw/{tag}/seed_{seed}.csv"
    logger = CSVLogger(log_path)

    train(env, agent, buffer, config, logger)

    logger.close()
    env.close()


def load_config(override):
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    for key in override:
        config[key].update(override[key])

    return config


if __name__ == "__main__":

    seeds = range(2)

    rho_values = [1, 4]

    # =========================
    # 1. Batch Size Sensitivity
    # =========================
    total_samples = [64, 128, 256, 512]

    for rho in rho_values:
        for total in total_samples:

            bs = max(total // rho, 8)

            override = {
                "train": {
                    "rho": rho,
                    "batch_size": bs
                }
            }

            config = load_config(override)
            tag = f"sens_bs_rho{rho}_bs{bs}"

            for seed in seeds:
                print(f"[Batch] rho={rho}, total={total}, bs={bs}, seed={seed}")
                run_experiment(config, seed, tag)

    # ===============================
    # 2. Target Update Sensitivity
    # ===============================
    target_updates = [250, 500, 1000, 2000, 4000]

    for rho in rho_values:
        for tu in target_updates:
            override = {
                "train": {
                    "rho": rho,
                    "target_update_freq": tu
                }
            }

            config = load_config(override)
            tag = f"sens_tu_rho{rho}_tu{tu}"

            for seed in seeds:
                print(f"[Target] rho={rho}, tu={tu}, seed={seed}")
                run_experiment(config, seed, tag)

    print("\n[INFO] Sensitivity experiments complete.")



# import torch
# import yaml

# from src.env.make_env import make_env
# from src.agents.dqn_agent import DQNAgent
# from src.replay.replay_buffer import ReplayBuffer
# from src.training.train import train
# from src.utils.logger import CSVLogger


# def run_experiment(config, seed, tag):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     env = make_env(
#         config["env"]["name"],
#         config["env"]["max_episode_steps"],
#         seed=seed
#     )

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n

#     agent = DQNAgent(state_dim, action_dim, config, device)
#     buffer = ReplayBuffer(state_dim, config["train"]["buffer_size"])

#     log_path = f"logs/raw/{tag}/seed_{seed}.csv"
#     logger = CSVLogger(log_path)

#     train(env, agent, buffer, config, logger)

#     logger.close()
#     env.close()


# def load_config(override):
#     with open("configs/default.yaml") as f:
#         config = yaml.safe_load(f)

#     for key in override:
#         config[key].update(override[key])

#     return config


# if __name__ == "__main__":

#     seeds = range(2)  # change to 15 later

#     rho_values = [1, 4]

#     # =========================
#     # 1. Batch Size Sensitivity
#     # =========================
#     base_batches = [32, 64, 128, 256]  # for rho=1

#     for rho in rho_values:
#         for base in base_batches:

#             bs = max(base // rho, 8)  # prevent too small batch

#             override = {
#                 "train": {
#                     "rho": rho,
#                     "batch_size": bs
#                 }
#             }

#             config = load_config(override)
#             tag = f"sens_bs_rho{rho}_bs{bs}"

#             for seed in seeds:
#                 print(f"[Batch] rho={rho}, bs={bs}, seed={seed}")
#                 run_experiment(config, seed, tag)
    
#     # ===============================
#     # 2. Target Update Sensitivity
#     # ===============================
#     target_updates = [250, 500, 1000, 2000, 4000]

#     for rho in rho_values:
#         for tu in target_updates:
#             override = {
#                 "train": {
#                     "rho": rho,
#                     "target_update_freq": tu
#                 }
#             }

#             config = load_config(override)
#             tag = f"sens_tu_rho{rho}_tu{tu}"

#             for seed in seeds:
#                 print(f"[Target] rho={rho}, tu={tu}, seed={seed}")
#                 run_experiment(config, seed, tag)