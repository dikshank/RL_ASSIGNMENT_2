# run_truncation.py

import torch
import yaml

from src.env.make_env import make_env
from src.agents.dqn_agent import DQNAgent
from src.replay.replay_buffer import ReplayBuffer
from src.training.train import train
from src.utils.logger import CSVLogger


def run_experiment(config, seed, tag):
    print(f"[INFO] Running {tag} | Seed {seed}")

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

    print(f"[INFO] Completed {tag} | Seed {seed}\n")


def load_config(base_path, override_dict):
    with open(base_path) as f:
        config = yaml.safe_load(f)

    for key in override_dict:
        config[key].update(override_dict[key])

    return config


if __name__ == "__main__":

    experiments = [
        ("trunc_200", {"env": {"max_episode_steps": 200}}),
        ("trunc_1000", {"env": {"max_episode_steps": 1000}}),
        ("trunc_2000", {"env": {"max_episode_steps": 2000}}),
    ]

    for tag, override in experiments:
        print(f"\n[INFO] Starting experiment: {tag}")
        config = load_config("configs/default.yaml", override)

        for seed in range(3):
            run_experiment(config, seed, tag)

    print("\n[INFO] All truncation experiments completed.")




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


# def load_config(base_path, override_dict):
#     with open(base_path) as f:
#         config = yaml.safe_load(f)

#     # Deep merge (safe override)
#     for key in override_dict:
#         config[key].update(override_dict[key])

#     return config


# if __name__ == "__main__":

#     experiments = [
#         ("trunc_200", {"env": {"max_episode_steps": 200}}),
#         ("trunc_1000", {"env": {"max_episode_steps": 1000}}),
#         ("trunc_2000", {"env": {"max_episode_steps": 2000}}),
#     ]

#     for tag, override in experiments:
#         config = load_config("configs/default.yaml", override)

#         for seed in range(2):  # start small
#             print(f"Running {tag}, seed {seed}")
#             run_experiment(config, seed, tag)