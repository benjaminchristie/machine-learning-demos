import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import GaussianNetwork, ReplayMemory, MLP
from networks import soft_update, hard_update
from torch.optim.adam import Adam

class SAC(nn.Module):
    def __init__(self, input_dim, output_dim, lr=3e-4):
        super(SAC, self).__init__()
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.lr = lr 
        self.hidden_dim = 128
        self.target_update_interval = 1
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # critic
        self.critic = MLP(self.input_dim, self.hidden_dim, 1).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr)
        self.critic_target = MLP(self.input_dim, self.hidden_dim, 1).to(self.device)
        hard_update(self.critic_target, self.critic) # copies weights
        
        # actor
        self.policy = GaussianNetwork(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr) 
        self.n_updates = 0
        return
    
    def select_action(self, state):
        action = self.policy.forward(state)
        return action
    
    def update_parameters(self, memory: ReplayMemory, batch_size=1, epochs=1):
        net_policy_loss = 0.0
        net_critic_loss = 0.0
        for _ in range(epochs):
            states, actions, rewards, next_states, dones = memory.sample(batch_size)
            states_t = torch.tensor(states, device=self.device, dtype=torch.float32)
            actions_t = torch.tensor(actions, device=self.device, dtype=torch.float32)
            rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            next_states_t = torch.tensor(next_states, device=self.device, dtype=torch.float32)
            mask_t = 1 - torch.tensor(dones, device=self.device, dtype=torch.float32)

            # train critic
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_states_t)
                qf1_next_target, qf2_next_target = self.critic_target(next_states_t, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = rewards_t + mask_t * self.gamma * min_qf_next_target
            qf1, qf2 = self.critic(states_t, actions_t)
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()
            
            # train actor
            pi, log_pi, _ = self.policy.sample(states_t)
            qf1_pi, qf2_pi = self.critic(states_t, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()
            
            if self.n_updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)
            self.n_updates += 1
            net_critic_loss += qf_loss.item()
            net_policy_loss += policy_loss.item()
            
        return net_policy_loss, net_critic_loss