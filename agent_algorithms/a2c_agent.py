import sys
import matplotlib.pyplot as plt
from utils.TimeBuffer import TimeBuffer
from networks.base_networks import *
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
    def __init__(self, env: gym.Env, actor=None, critic=None):
        assert isinstance(env, gym.Env)
        assert isinstance(env.action_space, gym.spaces.Discrete)

        self.actor = actor if actor else ACNetwork(env)
        self.critic = critic if critic else self.actor
        self.is_episodic = not hasattr(env, 'is_episodic') or (hasattr(env, 'is_episodic') and env.is_episodic)
        self.reward = 0
        self.testing_rewards = TimeBuffer(cfg.rewards_eval_window)
        self.average_rewards = []
        self.log_probs = []
        self.rewards = []
        self.value_estimates = []
        self.t = 0
        self.continuous_episode_length = cfg.pg.CONTINUOUS_EPISODE_LENGTH
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

        if episode_end or (not self.is_episodic and self.t == self.continuous_episode_length):
            discounted_rewards = [0]
            while self.rewards:
                # latest reward + (future reward * gamma)
                discounted_rewards.insert(0, self.rewards.pop() + (cfg.discount_factor * discounted_rewards[0]))
            discounted_rewards.pop(-1)  # remove the extra 0 placed before the loop

            Q_val = torch.Tensor(discounted_rewards).to(**args)
            Q_val = (Q_val - Q_val.mean()) / (Q_val.std() + 1e-9)
            V_val = torch.Tensor(self.value_estimates).to(**args)
            advantage = Q_val - V_val
            log_prob = torch.stack(self.log_probs)

            actor_loss = (-log_prob * advantage).mean()  # todo make sure this is elementwise product
            critic_loss = .5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            ac_loss.backward()

            self.optimizer.step()

            self.t = 0
            self.rewards = []
            self.log_probs = []

