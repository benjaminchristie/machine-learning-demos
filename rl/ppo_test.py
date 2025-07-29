from ppo import Env_ABC, PPO
import gymnasium as gym
import torch
import os
from argparse import ArgumentParser


class GymEnv(Env_ABC):
    def __init__(self, s: str):
        self.s = s
        self.env = gym.make(self.s)
        self.render_mode = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = self.env.observation_space.shape[0]  # type: ignore
        self.action_dim = self.env.action_space.shape[0]  # type: ignore

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        state, reward, terminated, truncated, _ = self.env.step(action.cpu().detach().numpy())
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
        return state, reward, terminated | truncated

    def reset(self):
        state, _ = self.env.reset()
        return torch.tensor(state, device=self.device, dtype=torch.float)

    def render(self):
        """toggles the render mode of the environment"""
        if self.render_mode is None:
            self.env = gym.make(self.s, render_mode="human")
            self.render_mode = "human"
        else:
            self.env = gym.make(self.s)
            self.render_mode = None


def save_model(model, path):
    os.makedirs("./weights/", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path):
    state_dict = torch.load(path, weights_only=False)
    model.load_state_dict(state_dict)


def main(args):
    env = GymEnv("Pendulum-v1")
    model = PPO(env.state_dim, 64, env.action_dim, env)
    if args.load:
        load_model(model, "./weights/ppo.pt")
    model.update_parameters(args.epochs, batch_size=1, max_timesteps_per_episode=300, n_updates_per_epoch=5)
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
    save_model(model, "./weights/ppo.pt")
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--load", default=False, action="store_true")
    main(parser.parse_args())
