import torch
import yaml

from src.env.make_env import make_env
from src.agents.dqn_agent_per import DQNAgentPER
from src.replay.per_buffer import PERBuffer
from src.training.train_per import train_per
from src.utils.logger import CSVLogger


def run(seed, rho):
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    config["train"]["rho"] = rho

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(config["env"]["name"], config["env"]["max_episode_steps"], seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgentPER(state_dim, action_dim, config, device)
    buffer = PERBuffer(state_dim, config["train"]["buffer_size"])

    log_path = f"logs/raw/per_rho_{rho}/seed_{seed}.csv"
    logger = CSVLogger(log_path)

    train_per(env, agent, buffer, config, logger)

    logger.close()
    env.close()


if __name__ == "__main__":
    for rho in [1, 2, 4, 8]:
        for seed in range(2):
            print(f"PER rho={rho}, seed={seed}")
            run(seed, rho)