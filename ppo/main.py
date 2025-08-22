import torch 
import matplotlib.pyplot as plt
from ppo import PPOEnv, PPO
import tqdm
from PPO import PPO as TheirPPO
import gymnasium as gym 
import panda_gym
import os 

class L2Env(PPOEnv):
    def __init__(self, device):
        self.render_mode = None
        self.device = device
        self.reset()
        
    def reset(self):
        """[0:2] = state [2:4] = goal

        Returns:
            _type_: _description_
        """
        self.state = torch.rand(4, device=self.device) * 20.0 - 10.0
        return self.state
    
    def render(self, *args, render_mode=None):
        self.render_mode = render_mode
        if self.render_mode is not None:
            plt.ion()
        else:
            plt.close()
            plt.ioff()
        
    def step(self, action: torch.Tensor):
        dist_before = torch.linalg.norm(self.state[2:4] - self.state[0:2])
        self.state[0:2] += action
        dist_after = torch.linalg.norm(self.state[2:4] - self.state[0:2])
        if self.render_mode is not None:
            plt.ion()
            plt.plot(*self.state[0:2].cpu().detach().numpy(), 'r.')
            plt.plot(*self.state[2:4].cpu().detach().numpy(), 'b.')
        reward = dist_before - dist_after
        return self.state, reward.unsqueeze(0), dist_before < 0.5
    
    
def save_model(model, path):
    os.makedirs("./weights/", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path):
    state_dict = torch.load(path, weights_only=False)
    model.load_state_dict(state_dict)

    
def main():
    env = L2Env(torch.device("cuda"))
    ppo = PPO(4, 32, 2, 0.001, env)
    tbar = tqdm.trange(100_000)
    for _ in tbar:
        l1, l2, dr, nr = ppo.update_parameters(timesteps=50)
        tbar.set_description(f"{l1:4.4f} {l2:4.4f} {dr:4.4f} {nr:4.4f}")
    tbar.close()
    return 

    
def main_theirs():
    import numpy as np
    theirs = TheirPPO(4, 2, 1e-3, 1e-3, 0.95, 5, 0.2, True, action_std_init=0.5)
    tbar = tqdm.trange(1000)
    render = False
    fig = plt.figure()
    plt.ion()
    for e in tbar:
        state = np.random.uniform(-10.0, 10.0, 4)
        net_reward = 0.0
        render = e % 100 == 0
        fig.clear()
        for _ in range(100):
            if render:
                plt.plot(*state[0:2], 'r.')
                plt.plot(*state[2:4], 'b.')
                plt.xlim([-10.0, 10.0])
                plt.ylim([-10.0, 10.0])
                plt.pause(0.01)
                
            action = theirs.select_action(state, eval=render)
            dist_before = np.linalg.norm(state[2:4] - state[0:2])
            dist_after = np.linalg.norm(state[2:4] - state[0:2] + action)
            r = dist_before - dist_after
            done = dist_before < 0.5
            if done:
                r = 100.0
            theirs.buffer.rewards.append(r)
            theirs.buffer.is_terminals.append(done)
            net_reward += r
            state[0:2] += action
            if done:
                break
        if render:
            print(net_reward)
        # if e % 100 == 0 and e != 0:
            # theirs.decay_action_std(0.01, 0.1)
        tbar.set_description(f"{net_reward:4.4f}")
        if len(theirs.buffer.actions) <= 2:
            continue
        theirs.update()
    tbar.close()
    return 


def main_panda_reach():
    import numpy as np
    env = gym.make("PandaReach-v3", reward_type="sparse", render_mode="human", control_type="joints")
    state, _ = env.reset()
    # state_dim = 6
    state_dim = 0
    for k, v in state.items():
        state_dim += len(v)
    action_dim = env.action_space.shape[0] # type: ignore
    ppo = TheirPPO(state_dim, action_dim, 3e-4, 3e-4, 0.95, 5, 0.2, True, 0.6)
    tbar = tqdm.trange(10000)
    plt.ion()
    rolling_rewards = []
    state_errors = []
    rolling_rewards_idx = 0
    print("State dim: ", state_dim)
    print("Action dim: ", action_dim)
    for e in tbar:
        # if e % 300 == 10:
        #     env = gym.make("PandaReach-v3", reward_type="sparse")
        # if e > 9000 or e % 300 == 0:
        #     env = gym.make("PandaReach-v3", reward_type="sparse", render_mode="human")
        finished = False
        t = 0
        T = 400
        net_reward = 0.0
        state, _ = env.reset()
        dist_before = np.linalg.norm(state["achieved_goal"] - state["desired_goal"])
        # state = np.concatenate((state["achieved_goal"], state["desired_goal"]))
        state = np.concatenate((state["observation"], state["achieved_goal"], state["desired_goal"]))
        while (not finished) and t < T:
            action = ppo.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            dist_after = np.linalg.norm(next_state["achieved_goal"] - next_state["desired_goal"])
            reward = dist_before - dist_after
            finished = done or truncated
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(finished)
            state = np.concatenate((next_state["observation"], next_state["achieved_goal"], next_state["desired_goal"]))
            t += 1
            net_reward += reward
            dist_before = dist_after
        state_error = np.linalg.norm(state[-4:-1] - state[-6:-3])
        if len(rolling_rewards) == 50:
            rolling_rewards[rolling_rewards_idx] = net_reward
            state_errors[rolling_rewards_idx] = state_error
            rolling_rewards_idx = (rolling_rewards_idx + 1) % 50
        else:
            rolling_rewards.append(net_reward)
            state_errors.append(state_error)
        plt.subplot(2, 1, 1)
        plt.plot(e, np.mean(rolling_rewards), 'r.')
        plt.title("net reward")
        plt.xlim([max(0, e - 1000), e])
        plt.subplot(2, 1, 2)
        plt.plot(e, np.mean(state_errors), 'b.')
        plt.title("final state error")
        plt.xlim([max(0, e - 1000), e])
        plt.pause(0.01)
        tbar.set_description(f"{net_reward:2.4f}")
        if t <= 1:
            print("tf")
            continue
        ppo.update()
        if e % 500 == 0 and e != 0:
            ppo.decay_action_std(0.05, 0.01)
        
        if e % 1000 == 0 and e != 0:
            save_model(ppo.policy, f"./weights/ppo_PandaReach-v3_{e}.pt")


    return

if __name__ == "__main__":
    main_panda_reach()
    # main_theirs()