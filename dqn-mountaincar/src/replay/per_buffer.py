import numpy as np


class PERBuffer:
    def __init__(self, state_dim, size, alpha=0.6):
        self.size = size
        self.ptr = 0
        self.full = False
        self.alpha = alpha

        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, 1), dtype=np.int64)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

        self.priorities = np.zeros((size,), dtype=np.float32)

    def add(self, s, a, r, s_next, done):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_next
        self.dones[self.ptr] = done

        self.priorities[self.ptr] = self.priorities.max() if self.ptr > 0 else 1.0

        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size):
        max_idx = self.size if self.full else self.ptr

        probs = self.priorities[:max_idx] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(max_idx, batch_size, p=probs)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
        )

    def update_priorities(self, indices, td_errors):
        self.priorities[indices] = np.abs(td_errors) + 1e-6