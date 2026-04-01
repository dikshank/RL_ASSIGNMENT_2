import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, size):
        self.size = size
        self.ptr = 0
        self.full = False

        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, 1), dtype=np.int64)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

    def add(self, s, a, r, s_next, done):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_next
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size):
        max_idx = self.size if self.full else self.ptr
        idx = np.random.randint(0, max_idx, size=batch_size)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )