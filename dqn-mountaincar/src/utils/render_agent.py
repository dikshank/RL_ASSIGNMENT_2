import torch
import yaml
import time
import os

from src.env.make_env import make_env
from src.agents.dqn_agent import DQNAgent


def load_config():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


def run_render(seed=0, episodes=3):
    print("[INFO] Loading configuration...")
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Creating environment with rendering...")
    env = make_env(
        config["env"]["name"],
        config["env"]["max_episode_steps"],
        seed=seed,
        render_mode="human"
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("[INFO] Initializing agent...")
    agent = DQNAgent(state_dim, action_dim, config, device)

    model_path = "models/dqn_model.pth"
    assert os.path.exists(model_path), "[ERROR] Model file not found!"

    print(f"[INFO] Loading trained model from {model_path}")
    agent.q_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.q_net.eval()

    # Ensure no exploration during evaluation
    agent.epsilon = 0.0

    print("[INFO] Starting rendering...\n")

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"[INFO] Episode {ep} started")

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action = agent.q_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            total_reward += reward
            step += 1

            state = next_state

            time.sleep(0.02)  # control speed for visualization

        print(f"[RESULT] Episode {ep} | Return: {total_reward:.2f} | Steps: {step}")

    print("\n[INFO] Rendering complete.")
    env.close()


if __name__ == "__main__":
    run_render()


# import torch
# import yaml
# import time
# import numpy as np

# from src.env.make_env import make_env
# from src.agents.dqn_agent import DQNAgent


# def load_config():
#     with open("configs/default.yaml") as f:
#         return yaml.safe_load(f)


# def normalize_state(state):
#     low = np.array([-1.2, -0.07])
#     high = np.array([0.6, 0.07])
#     return (state - low) / (high - low)


# def run_render(seed=0, episodes=3):
#     config = load_config()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     env = make_env(
#         config["env"]["name"],
#         config["env"]["max_episode_steps"],
#         seed=seed,
#         render_mode="human"
#     )

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n

#     agent = DQNAgent(state_dim, action_dim, config, device)

#     agent.q_net.load_state_dict(torch.load("models/dqn_model.pth", map_location=device))
#     agent.q_net.eval()

#     for ep in range(episodes):
#         state, _ = env.reset()
#         done = False
#         total_reward = 0

#         while not done:
#             state_norm = normalize_state(state)  # ✅ FIX
#             state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)

#             with torch.no_grad():
#                 action = agent.q_net(state_tensor).argmax().item()

#             next_state, reward, terminated, truncated, _ = env.step(action)

#             done = terminated or truncated
#             total_reward += reward

#             state = next_state

#             time.sleep(0.02)

#         print(f"Episode {ep}: Return = {total_reward}")

#     env.close()


# if __name__ == "__main__":
#     run_render()