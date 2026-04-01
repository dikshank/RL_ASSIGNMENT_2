import torch
import yaml
import os
import numpy as np
import random
import time

from src.env.make_env import make_env
from src.agents.dqn_agent import DQNAgent
from src.replay.replay_buffer import ReplayBuffer
from src.training.train import train
from src.utils.logger import CSVLogger


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        print('GPU available...')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run(seed=0, exp_id=None):
    set_seed(seed)

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

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

    log_path = f"logs/raw/baseline/seed_{seed}.csv"
    logger = CSVLogger(log_path)

    print(f"Running seed {seed}")

    train(env, agent, buffer, config, logger)

    logger.close()
    env.close()


if __name__ == "__main__":
    exp_id = int(time.time())

    for seed in range(3):
        run(seed, exp_id)