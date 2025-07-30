import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import MLP
from torch.optim.adamw import AdamW
from torch.distributions import MultivariateNormal
import tqdm

from abc import ABC, abstractmethod
from typing import Tuple


class Env_ABC(ABC):
    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """steps the environment state according to the action tensor

        Args:
            action (torch.Tensor): action tensor (batch size N = 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, bool]: next_state, reward, terminated
        """
        pass

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """resets the environment to a random (valid) configuration

        Returns:
            torch.Tensor: state
        """
        pass

    @abstractmethod
    def render(self):
        pass


class PPO(nn.Module):
    """
    PPO is designed to be on-policy.
    We embed the environment step function
    in the learning algorithm here,
    so a reference to the env must be passed
    to the constructor.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, env: Env_ABC):
        super(PPO, self).__init__()
        self.gamma = 0.95
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 5e-3
        self.actor = MLP(input_dim, hidden_dim, output_dim, n_hidden_layers=3).to(self.device)
        self.critic = MLP(input_dim, hidden_dim, 1, n_hidden_layers=3).to(self.device)
        self.actor_optim = AdamW(self.actor.parameters(), self.lr)
        self.critic_optim = AdamW(self.critic.parameters(), self.lr)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cov_var = torch.full(size=(output_dim,), fill_value=0.5, device=self.device)
        self.cov_mat = nn.Parameter(torch.diag(self.cov_var), requires_grad=False)
        self.env = env
        
    def train(self, mode: bool = True):
        super(PPO, self).train(mode)
        self.cov_var = torch.full(size=(self.output_dim,), fill_value=0.5, device=self.device)
        self.cov_mat = nn.Parameter(torch.diag(self.cov_var), requires_grad=False)
        return self
    
    def eval(self):
        super(PPO, self).eval()
        self.cov_var = torch.full(size=(self.output_dim,), fill_value=0.001, device=self.device)
        self.cov_mat = nn.Parameter(torch.diag(self.cov_var), requires_grad=False)
        return self
        

    def forward(self, x):
        mean = self.actor.forward(x)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(self, states, actions):
        V = self.critic(states).squeeze()
        return V, self.get_log_probs(states, actions)

    def get_log_probs(self, state, action):
        mean = self.actor.forward(state)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(action)
        return log_prob

    def compute_ratings(self, rewards_t: torch.Tensor):
        batch_rtgs = []
        for ep_rews in reversed(rewards_t):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)
        return batch_rtgs

    @torch.no_grad()
    def rollout(self, batch_size: int = 1, max_timesteps: int = 200):
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_ratings = []
        curr_batch_size = 0
        while curr_batch_size < batch_size:
            ep_rews = []
            state = self.env.reset()
            terminated = False
            t = 0
            for _ in range(max_timesteps):
                t += 1
                batch_states.append(state)
                action, log_prob = self.forward(state)
                state, reward, terminated = self.env.step(action)
                ep_rews.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                if terminated:
                    break
            batch_rewards.append(torch.concat(ep_rews))
            curr_batch_size += 1
        batch_states = torch.stack(batch_states)
        batch_actions = torch.stack(batch_actions)
        batch_log_probs = torch.stack(batch_log_probs)
        batch_ratings = self.compute_ratings(torch.stack(batch_rewards))

        return batch_states, batch_actions, batch_log_probs, batch_ratings, batch_rewards

    def update_parameters(self, epochs: int = 1, batch_size: int = 1, max_timesteps_per_episode: int = 200, n_updates_per_epoch: int = 5):
        EPSILON = 0.5
        N = n_updates_per_epoch
        tbar = tqdm.trange(epochs)
        for _ in tbar:
            states, actions, log_probs, ratings, batch_rewards = self.rollout(batch_size, max_timesteps=max_timesteps_per_episode)
            if len(states) <= 1:
                continue
            V, _ = self.evaluate(states, actions)
            A_k = ratings.detach() - V.detach()
            # normalizing the advantage helps with convergence
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            avg_actor_loss = 0.0
            avg_critic_loss = 0.0
            for _ in range(N):
                V, curr_log_probs = self.evaluate(states, actions)
                ratios = torch.exp(curr_log_probs - log_probs.detach())
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - EPSILON, 1 + EPSILON) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = F.mse_loss(V, ratings)
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                avg_actor_loss += actor_loss.item()
                avg_critic_loss += critic_loss.item()

            tbar.set_description(
                f"Actor Loss {avg_actor_loss/N:4.4f} Critic Loss {avg_critic_loss/N:3.3f} Ratings {torch.mean(ratings[..., 0]):3.3f} Rewards: {sum([torch.sum(x) for x in batch_rewards]):3.3f}"
            )
