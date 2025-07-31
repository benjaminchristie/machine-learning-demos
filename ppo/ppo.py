import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch.distributions import MultivariateNormal
import tqdm

from typing import Union, Tuple
from abc import ABC, abstractmethod



def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)   
        
def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    return

class PPOEnv(ABC):
    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """steps the environment state according to the action tensor

        Args:
            action (torch.Tensor): action tensor (batch size N = 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, bool]: next_state, reward, terminated
        """

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """resets the environment to a random (valid) configuration

        Returns:
            torch.Tensor: state
        """

    @abstractmethod
    def render(self, *args, **kwargs) -> None:
        """sets the rendering mode of the environment

        Returns:
            None
        """


class PPO(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float,
        env: PPOEnv,
        device: Union[str, None] = None,
    ):
        """Note that PPO is an on-policy RL algorithm. We pass a reference
        to the environment here to emphasize that

        Args:
            input_dim (int): state_dim
            hidden_dim (int): hidden_dim
            output_dim (int): action_dim
            lr (float): learning_rate
            env (PPOEnv): environment to train on
            max_interval_bw_updates (int, optional): Defaults to 5.
            max_timesteps_per_episode (int, optional):  Defaults to 1000.
            device (Union[str, None], optional): Defaults to "cuda" if None.
        """
        super(PPO, self).__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ).to(self.device)
        
        self.old_actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)
        
        hard_update(self.old_actor, self.actor)

        self.actor_optim = AdamW(self.actor.parameters(), lr)
        self.critic_optim = AdamW(self.critic.parameters(), lr)

        self.state_dim = input_dim
        self.action_dim = output_dim
        self.env = env

        self.cov_var = torch.full(size=(output_dim,), fill_value=0.5, device=self.device, requires_grad=False)
        self.cov_mat = torch.diag(self.cov_var)
        
        self.gamma = 0.95
        self.alpha = 0.2
        self.epsilon = 0.2
        
        self.apply(_init_weights)
        
    def decay_cov(self, ratio: float = 0.9):
        self.cov_mat *= ratio
        return

    def train(self, mode: bool = True):
        super(PPO, self).train(mode)
        self.is_training = mode
        return self

    def eval(self):
        super(PPO, self).eval()
        self.is_training = False
        return self

    def forward(self, x: torch.Tensor, use_old=True):
        if use_old:
            mean = self.old_actor(x)
        else:
            mean = self.actor(x)
        dist = MultivariateNormal(mean, self.cov_mat)
        return dist

    def evaluate(self, states, actions):
        V = self.critic(states).squeeze()
        dist = self.forward(states)
        log_probs = dist.log_prob(actions)
        return V, log_probs

    @torch.no_grad()
    def rollout(self, max_timesteps: int = 1000):
        T = 0
        states = []
        actions = []
        rewards = []
        log_probs = []
        state = self.env.reset()
        for t in range(max_timesteps):
            dist = self.forward(state)
            action = dist.rsample()
            log_prob = dist.log_prob(action)
            next_state, reward, done = self.env.step(action)
            states.append(state.clone())
            actions.append(action)
            log_probs.append(log_prob)            
            rewards.append(reward)
            T = t + 1
            if done:
                break
            state = next_state
        
        return T, torch.vstack(states), torch.vstack(actions), torch.vstack(rewards), torch.vstack(log_probs)
    
    def update_parameters(self, epochs: int = 1, timesteps: int = 20):
        T, states, actions, rewards, old_log_probs = self.rollout(timesteps)
        if T == 1:
            return 0.0, 0.0, 0.0, 0.0
        discounted_rewards = torch.empty(T, device=self.device)
        discounted_reward = 0.0
        for t in range(T - 1, -1, -1):
            discounted_rewards[t] = rewards[t] + self.gamma * discounted_reward
            discounted_reward += discounted_rewards[t]
        # print(T, states, actions, rewards, old_log_probs, discounted_rewards)
            
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
        # print(discounted_rewards)
        old_state_values = self.critic(states)
        A = discounted_rewards.detach() - old_state_values.detach()
        net_actor_loss = 0.0
        net_critic_loss = 0.0
        for _ in range(epochs):
            dist = self.forward(states, use_old=False)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            values = self.critic(states)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            # print(ratios.shape, log_probs.shape, old_log_probs.shape, values.shape, discounted_rewards.shape)            
            surr1 = ratios * A
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * A
            
            # print(surr1, surr2, A, surr1.shape, surr2.shape)
            # input()
            
            actor_loss = (-torch.min(surr1, surr2) - 0.01 * entropy).mean()
            
            critic_loss = F.mse_loss(values.squeeze(), discounted_rewards)
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            
            net_actor_loss += actor_loss.item()
            net_critic_loss += critic_loss.item()

        hard_update(self.old_actor, self.actor)
            
        return net_actor_loss / epochs, net_critic_loss / epochs, discounted_rewards[-1].item(), torch.sum(rewards).item()