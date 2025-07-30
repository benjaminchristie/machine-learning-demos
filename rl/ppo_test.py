from ppo import Env_ABC, PPO
import gymnasium as gym
import panda_gym
from panda_gym.envs.core import Task
from panda_gym.utils import distance
import numpy as np
import torch
import os
from argparse import ArgumentParser


class GymEnv(Env_ABC):
    def __init__(self, s: str, **env_kwargs):
        self.s = s
        self.kwargs = env_kwargs
        self.env = gym.make(self.s, **self.kwargs)
        self.render_mode = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.discrete = False 
            self.state_dim = self.env.observation_space.shape[0]  # type: ignore
            self.action_dim = self.env.action_space.shape[0]  # type: ignore
        except IndexError:
            self.discrete = True
            self.state_dim = self.env.observation_space.shape[0]  # type: ignore
            self.action_dim = self.env.action_space.n # type: ignore
        except TypeError: # 
            self.discrete = False
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                self.state_dim = 0
                self.action_dim = self.env.action_space.shape[0] # type: ignore
                for _, v in self.env.observation_space.items():
                    self.state_dim += v.shape[0] # type: ignore
            else:
                raise NotImplementedError
        print(f"State_dim: {self.state_dim} Action_dim: {self.action_dim}")
                
    @torch.no_grad()
    def step(self, action: torch.Tensor):
        if self.discrete:
            discrete_action = torch.argmax(action.cpu()).item()
            state, reward, terminated, truncated, _ = self.env.step(discrete_action)
        else:
            state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        return state, reward, terminated | truncated

    def reset(self):
        state, _ = self.env.reset()
        return torch.tensor(state, device=self.device, dtype=torch.float)

    def render(self):
        """toggles the render mode of the environment"""
        if self.render_mode is None:
            self.env = gym.make(self.s, render_mode="human", **self.kwargs)
            self.render_mode = "human"
        else:
            self.env = gym.make(self.s, **self.kwargs)
            self.render_mode = None
            
class PandaEnv(GymEnv):
    def __init__(self, s: str):
        super().__init__(s, reward_type="dense")
        
    @torch.no_grad()
    def step(self, action: torch.Tensor):
        state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
        state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"]))
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        return state, reward, terminated | truncated
    
    def reset(self):
        state, _ = self.env.reset()
        state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"]))
        return torch.tensor(state, device=self.device, dtype=torch.float)


def save_model(model, path):
    os.makedirs("./weights/", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path):
    state_dict = torch.load(path, weights_only=False)
    model.load_state_dict(state_dict)


def main(args):
    if "Panda" in args.env:
        env = PandaEnv(args.env)
    else:
        env = GymEnv(args.env)
    model = PPO(env.state_dim, 1024, env.action_dim, env)
    filename = f"./weights/ppo_{args.env}.pt"
    if args.load and os.path.exists(filename):
        load_model(model, filename)
    model.update_parameters(args.epochs, batch_size=1, max_timesteps_per_episode=500, n_updates_per_epoch=5)
    env.render()
    n_games = 10
    for game in range(n_games):
        net_reward = 0.0
        state = env.reset()
        for _ in range(300):
            action, _ = model.forward(state)
            state, reward, _ = env.step(action)
            net_reward += reward.item()
        print(f"Game {game} Net reward: {net_reward:2.2f}")
    save_model(model, filename)
    return

def render_test():
    import matplotlib.pyplot as plt
    import tqdm
    env = PandaEnv("PandaReach-v3")
    seed = 123
    # figure = plt.figure()
    # ax = figure.add_subplot(projection="3d")
    X = np.linspace(-2.0, 2.0, 50)
    Y = np.linspace(-2.0, 2.0, 50)
    Xgrid, Ygrid = np.meshgrid(X, Y)
    Z = np.empty(Xgrid.shape)
    for i in tqdm.trange(Xgrid.shape[0]):
        for j in tqdm.trange(Xgrid.shape[1], leave=False):
            env.env.reset(seed=seed)
            u = np.array([Xgrid[i, j], 0.0, Ygrid[i, j]])
            _, reward, _ = env.step(torch.tensor(u))
            Z[i, j] = reward[0].cpu().item()
    # for _ in tqdm.tra/nge(100):
        # env.env.reset/(seed=seed)
        # u = env.env.a/ction_space.sample()
        # _, reward, _ /= env.step(torch.tensor(u))
        # ax.scatter(/*u[0:2], reward[0].cpu().item(), marker='.')
        # X.append(u[0]/)
        # Y.append(u[1]/)
        # Z.append(rewa/rd[0].cpu().item())
    cs = plt.contourf(X, Y, Z, 
                  hatches =['-', '/',
                            '\\', '//'],
                  cmap ='Greens',
                  extend ='both',
                  alpha = 1)

    plt.colorbar(cs)

    plt.show()
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--load", default=False, action="store_true")
    main(parser.parse_args())
    # render_test()
