from ppo import PPO, Env_ABC
import torch
import matplotlib.pyplot as plt
import os 

class ReacherEnv(Env_ABC):
    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_goal_range = torch.FloatTensor([-10.0, -10.0, -10.0]).to(self.device)
        self.max_goal_range = torch.FloatTensor([10.0, 10.0, 10.0]).to(self.device)
        self.state_dim = 6
        self.action_dim = 3
        self.reward_normalizer = torch.inf
        self.render_mode = None

    def reset(self):
        self.state = torch.rand(3).to(self.device) * 20.0 - 10.0
        self.goal = torch.rand(3).to(self.device) * (self.max_goal_range - self.min_goal_range) + self.min_goal_range
        if self.render_mode == "human":
            plt.close()
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(projection="3d")
        self.reward_normalizer = torch.linalg.norm(self.state - self.goal)
        return torch.concat((self.state, self.goal))
    
    @torch.no_grad()
    def dynamics(self, u: torch.Tensor):
        # dx0 = torch.sin(u[0]) * torch.exp(torch.cos(u[2]))
        # dx1 = torch.cos(u[1]) * torch.exp(- u[0] * torch.cos(u[2] * u[0]))
        # dx2 = torch.sin(u[2]) * torch.exp(u[2] * torch.cos(u[1]))
        dx0 = u[0]
        dx1 = u[1]
        dx2 = u[2]
        self.state[0] += dx0 
        self.state[1] += dx1 
        self.state[2] += dx2
        return self.state

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        state = self.dynamics(action)
        state = torch.concat((state, self.goal))
        reward = -torch.linalg.norm(state[0:3] - state[3:6]).unsqueeze(0)
        if self.render_mode == "human":
            self.ax.scatter(*(state[0:3].cpu().numpy()), c="r", marker='.')
            self.ax.scatter(*(state[3:6].cpu().numpy()), c="b", marker='.')
            plt.pause(0.01)
        return state, reward / self.reward_normalizer, reward >= -1.0
    
    def render(self):
        """toggles the render mode of the environment"""
        if self.render_mode is None:
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(projection="3d")
            self.render_mode = "human"
        else:
            plt.close()
            plt.ioff()
            self.render_mode = None
            
def save_model(model, path):
    os.makedirs("./weights/", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path):
    state_dict = torch.load(path, weights_only=False)
    model.load_state_dict(state_dict)            
            
def main():
    load = False
    env = ReacherEnv()
    model = PPO(env.state_dim, 16, env.action_dim, env)
    filename = "./weights/ppo_test2.pt"
    if load and os.path.exists(filename):
        load_model(model, filename)
    model.train()
    model.update_parameters(1000, batch_size=1, max_timesteps_per_episode=50, n_updates_per_epoch=5)
    model.eval()
    env.render()
    n_games = 10
    for game in range(n_games):
        net_reward = 0.0
        state = env.reset()
        for _ in range(50):
            action, _ = model.forward(state)
            state, reward, done = env.step(action)
            net_reward += reward.item()
            if done:
                break
        print(f"Game {game} Net reward: {net_reward:2.2f}")
    save_model(model, filename)
    return 

if __name__ == "__main__":
    main()