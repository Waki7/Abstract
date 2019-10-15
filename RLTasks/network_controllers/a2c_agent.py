import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import gym
from utils.TimeBuffer import TimeBuffer
import RLTasks.config as cfg
from RLTasks.network_controllers.base_networks import *
from tensorboardX import SummaryWriter

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
type = torch.float
args = {'device': device, 'dtype': type}


class A2CAgent():
    # this agent can work with environments x, y, z (life and gym envs)
    # try to make the encoding part separate
    def __init__(self, env: gym.Env, model = None):
        assert isinstance(env, gym.Env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        if model:
            print('fas')
        else:
            print('asldkfj')

        self.model = model if model else ACNetwork(env)
        self.is_episodic = not hasattr(env, 'is_episodic') or (hasattr(env, 'is_episodic') and env.is_episodic)
        self.reward = 0
        self.testing_rewards = TimeBuffer(cfg.rewards_eval_window)
        self.average_rewards = []
        self.log_probs = []
        self.rewards = []
        self.value_estimates = []
        self.t = 0
        self.optimizer = getattr(torch.optim, cfg.gym.OPTIMIZER)(self.model.parameters(), lr=cfg.gym.LR)
        self.writer = SummaryWriter()

    def step(self, env_input):
        action, log_prob, value_estimate = self.model.get_action(env_input)
        self.value_estimates.append(value_estimate)
        self.log_probs.append(log_prob)
        self.t += 1
        return action

    def update_policy(self, env_reward, episode_end):
        self.rewards.append(env_reward)
        if episode_end or (not self.is_episodic and self.t == cfg.pg.CONTINUOUS_EPISODE_LENGTH):
            discounted_rewards = [0]
            while self.rewards or self.value_estimates:
                # latest reward + (future reward * gamma)
                discounted_rewards.insert(0, self.rewards.pop() + (cfg.discount_factor * discounted_rewards[0]))
            discounted_rewards.pop(-1)  # remove the extra 0 placed before the loop

            Q_val = torch.tensor(discounted_rewards)
            V_val = torch.Tensor(self.value_estimates)
            advantage = Q_val - V_val
            log_prob = torch.stack(self.log_probs)

            actor_loss = (-log_prob * advantage).mean() #todo make sure this is elementwise product
            critic_loss = .5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss
            # policy_gradient = []
            # for log_prob, Gt in zip(self.log_probs, discounted_rewards):
            #     policy_gradient.append(log_prob * Gt)

            self.optimizer.zero_grad()
            ac_loss.backward()
            self.optimizer.step()

            self.t = 0
            self.rewards = []
            self.log_probs = []

    def log_predictions(self, writer=sys.stdout):
        writer.write('\nAgent Summary at timestep ' + str(self.t) + '\n')
        writer.write('prediction to environment: ' + str(self.model.get_env_pred_val()) + '\n')
        writer.write(str(self.pred_val) + ', ' + str(self.pred_feel_val))
        writer.write('\n\n full reward is: ' + str(self.reward))
        writer.write('\n')
        writer.flush()

    def store_results(self, reward):
        self.testing_rewards.insert(self.t, reward)
        if not self.t % cfg.rewards_eval_window:
            self.average_rewards.append(np.average(self.testing_rewards.getData()))

    def plot_results(self):
        if cfg.results_path is not None:
            print('trying')
            print(self.average_rewards)
            print(cfg.results_path)
            plt.plot(self.average_rewards)
            plt.savefig(cfg.results_path + 'averageRewards.png')