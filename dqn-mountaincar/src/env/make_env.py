import gymnasium as gym

def make_env(env_name, max_episode_steps, seed=None):
    env = gym.make(env_name)

    if max_episode_steps is not None:
        env._max_episode_steps = max_episode_steps  # override truncation

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env