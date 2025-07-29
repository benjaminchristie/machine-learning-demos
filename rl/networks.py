import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Normal
import numpy as np


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    return

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)    
    

class ReplayMemory:
    def __init__(self, capacity=1000):
        assert capacity > 0
        self.capacity = int(capacity)
        self.position = 0
        self.size = 0
        self.buffer = np.zeros(self.capacity, dtype=tuple)

    def push(self, *args):
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        batch = np.random.choice(self.buffer[0 : self.size], batch_size)
        args = map(np.stack, zip(*batch))
        return args

    def __len__(self):
        return self.size

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_hidden_layers: int = 2):
        super(MLP, self).__init__()
        layers = []
        
        if n_hidden_layers == 0 or hidden_dim == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.ParameterList(layers)
        self.activation = F.relu
        self.final_activation = nn.Identity()
        
        self.apply(_init_weights)
        
    def forward(self, x: torch.Tensor):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.final_activation(self.layers[-1](x))
        return x
    
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.q1 = MLP(num_inputs + num_actions, hidden_dim, 1, 2)
        self.q2 = MLP(num_inputs + num_actions, hidden_dim, 1, 2)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = self.q1(xu)
        x2 = self.q2(xu)
        return x1, x2    
    

class GaussianNetwork(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, n_hidden_layers: int = 1, scale=1.0, bias=0.0, deterministic=False
    ):
        super(GaussianNetwork, self).__init__()
        self.scale = scale
        self.bias = bias
        self.epsilon = 1e-6
        self.log_sig_min = -20
        self.log_sig_max = 2
        self.mlp = MLP(input_dim, hidden_dim, hidden_dim, n_hidden_layers)
        self.mu1 = nn.Linear(hidden_dim, hidden_dim)
        self.mu2 = nn.Linear(hidden_dim, output_dim)
        self.std1 = nn.Linear(hidden_dim, hidden_dim)
        self.std2 = nn.Linear(hidden_dim, output_dim)
        self.mlp.activation = nn.LeakyReLU(negative_slope=0.2)
        self.mlp.final_activation = nn.LeakyReLU(negative_slope=0.2) # type: ignore
        self.deterministic = deterministic

    def forward(self, x):
        x = self.mlp.forward(x)
        mu = x
        mu = self.mlp.activation(self.mu1(x))
        mu = self.mu2(x)
        std = self.mlp.final_activation(self.std1(x))
        std = self.std2(x)
        std = torch.clamp(std, min=self.log_sig_min, max=self.log_sig_max)
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        if not self.deterministic:
            std = std.exp()
            normal = Normal(mu, std)
            x_t = normal.rsample()
            y_t = self.scale * F.sigmoid(x_t) + self.bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.scale * (1 - y_t.pow(2)) + self.epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mu = self.scale * torch.sigmoid(mu) + self.bias
            return y_t, log_prob, mu
        else:
            x_t = mu
            y_t = self.scale * F.sigmoid(x_t) + self.bias
            return y_t, 0.0, y_t
