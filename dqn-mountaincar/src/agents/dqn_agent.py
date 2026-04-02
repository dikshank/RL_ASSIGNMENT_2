import torch
import torch.nn.functional as F
import numpy as np
from .networks import QNetwork

class DQNAgent:
    def __init__(self, state_dim, action_dim, config, device):
        self.device = device
        self.gamma = config["train"]["gamma"]

        hidden_dim = config["network"]["hidden_dim"]

        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config["train"]["lr"])

        self.epsilon = config["exploration"]["epsilon_start"]
        self.epsilon_end = config["exploration"]["epsilon_end"]
        self.epsilon_decay_steps = config["exploration"]["epsilon_decay_steps"]
        self.epsilon_decay = (self.epsilon - self.epsilon_end) / self.epsilon_decay_steps

        self.action_dim = action_dim
        self.train_steps = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), q_values.mean().item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)