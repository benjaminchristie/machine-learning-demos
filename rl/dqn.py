import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import GaussianNetwork, ReplayMemory
from networks import soft_update
from torch.optim.adam import Adam


class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_dim, scale=1.0, bias=0.0, lr=1e-3, device=None):
        super(DQN, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.eval_net = GaussianNetwork(
            n_inputs,
            n_outputs,
            hidden_dim,
            scale=scale,
            bias=bias,
            deterministic=False,
        ).to(self.device)
        self.target_net = GaussianNetwork(
            n_inputs,
            n_outputs,
            hidden_dim,
            scale=scale,
            bias=bias,
            deterministic=False,
        ).to(self.device)
        self.action_dim = n_outputs
        self.epsilon = 0.1
        self.gamma = 0.99
        self.tau = 0.005
        self.optim = Adam(self.eval_net.parameters(), lr)

    def sample(self, state):
        return self.eval_net.sample(state)

    def forward(self, state):
        action, _, _ = self.sample(state)
        action_hot = F.one_hot(torch.argmax(action, dim=1), num_classes=self.action_dim)
        return action_hot

    def select_action(self, state, epsilon: float | None = None):
        if epsilon is None:
            epsilon = self.epsilon
        if epsilon > torch.rand(1):
            action_value = torch.rand(self.action_dim).to(self.device)
        else:
            action_value, _, _ = self.eval_net.sample(state)
        action = torch.argmax(action_value)
        return action

    def update_parameters(
        self,
        memory: ReplayMemory,
        epochs: int,
        batch_size: int,
        device=None,
        normalize=True,
        soft_update_rate=2,
    ):
        """
        memory should be a replaymemory buffer with
        [states, actions, rewards, next_states, mask]
        """
        if device is None:
            device = self.device
        net_loss = 0.0
        lossf = F.smooth_l1_loss
        for ep in range(epochs):
            states, actions, rewards, next_state_batch, mask = memory.sample(batch_size)
            states_t = torch.tensor(states, device=device, dtype=torch.float32)
            states_p1_t = torch.tensor(next_state_batch, device=device, dtype=torch.float32)
            actions_t = torch.tensor(actions, device=device, dtype=torch.int64)
            rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
            mask_t = torch.tensor(mask, device=device, dtype=torch.float32)

            mu_eval, _ = self.eval_net(states_t)
            q_eval = mu_eval.gather(1, actions_t)
            mu_next, _ = self.target_net(states_p1_t)
            q_next = mu_next.detach()
            if normalize:
                q_target = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-7) + self.gamma * mask_t * q_next.max(1)[0].view(batch_size, 1)
            else:
                q_target = rewards_t + self.gamma * mask_t * q_next.max(1)[0].view(batch_size, 1)
            loss = lossf(q_eval, q_target)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            net_loss += loss.item()
            if ep % soft_update_rate == 0:
                soft_update(self.eval_net, self.target_net, self.tau)
        return net_loss / epochs

