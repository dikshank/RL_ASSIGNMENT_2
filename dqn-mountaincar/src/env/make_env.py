import gymnasium as gym


def make_env(env_name, max_episode_steps, seed=None, render_mode=None):
    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode  # ✅ add this
    )

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env