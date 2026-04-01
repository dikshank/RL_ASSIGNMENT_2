import numpy as np
from tqdm import tqdm


def normalize_state(state):
    low = np.array([-1.2, -0.07])
    high = np.array([0.6, 0.07])
    return (state - low) / (high - low)


def train_per(env, agent, buffer, config, logger):
    total_steps = config["train"]["total_timesteps"]
    batch_size = config["train"]["batch_size"]
    min_buffer = config["train"]["min_buffer_size"]
    target_freq = config["train"]["target_update_freq"]
    rho = config["train"]["rho"]

    state, _ = env.reset()
    state = normalize_state(state)

    episode_return = 0
    episode_length = 0
    episode = 0

    for t in tqdm(range(total_steps)):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        next_state_norm = normalize_state(next_state)

        buffer.add(state, action, reward, next_state_norm, done)

        state = next_state_norm
        episode_return += reward
        episode_length += 1

        loss, mean_q = 0, 0

        if buffer.ptr > min_buffer:
            for _ in range(rho):
                batch = buffer.sample(batch_size)
                loss, mean_q = agent.update(batch, buffer)

        if t % target_freq == 0:
            agent.update_target()

        agent.decay_epsilon()

        if done:
            logger.log([
                episode, t, episode_return, episode_length,
                agent.epsilon, loss, mean_q
            ])

            state, _ = env.reset()
            state = normalize_state(state)

            episode_return = 0
            episode_length = 0
            episode += 1