from agent_controllers.factory import register_controller
from agent_algorithms.cra_agent import *
import logging

from networks.base_networks import *
import agent_algorithms

from shutil import copy as copy_file

import numpy as np
import os
import random
import torch
import config as cfg


class BaseController():
    def __init__(self):
        self.t = 0

    def store_results(self, reward):
        self.testing_rewards.insert(self.t, reward)
        if not self.t % cfg.rewards_eval_window:
            self.average_rewards.append(np.average(self.testing_rewards.getData()))

@register_controller
class SingleAgentController():
    def __init__(self, env, cfg):
        self.cfg = cfg
        self.env = env
        self.agents = self.make_agents()

    def make_agents(self):
        return self.cfg['agents']

    def teach_agents(self):

        is_episodic = not hasattr(self.env, 'is_episodic') or (
                    hasattr(self.env, 'is_episodic') and self.env.is_episodic)
        update_rate = 1 if is_episodic else cfg.experiment.UPDATE_RATE
        max_episodes = cfg.experiment.MAX_EPISODES if is_episodic else 1
        all_rewards = []
        numsteps = []
        avg_numsteps = []

        for episode in range(max_episodes):
            state = self.env.reset()
            rewards = []
            step = 0
            while True:
                action = self.agent.step(state)
                state, reward, episode_end, _ = self.env.step(action)
                rewards.append(reward)
                self.agent.update_policy(reward, episode_end)

                if (is_episodic and episode_end) or (not is_episodic and step % update_rate == 0):
                    numsteps.append(step)
                    avg_numsteps.append(np.mean(numsteps[-cfg.experiment.EVAL_REWARDS_WINDOW:]))
                    all_rewards.append(np.sum(rewards))
                    logging.debug(
                        "episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(
                            np.sum(rewards), decimals=3), np.round(
                            np.mean(all_rewards[-cfg.experiment.EVAL_REWARDS_WINDOW:]), decimals=3), step))
                    if is_episodic:
                        break
                    else:
                        rewards = []
                if step > cfg.experiment.MAX_STEPS:
                    break
                step += 1

        plot_rewards = [np.mean(all_rewards[i:i + 50]) for i in range(0, len(all_rewards), 50)]
        env_id = (self.env.unwrapped.spec.id if not hasattr(self.env, 'envs') else env.envs[0].unwrapped.spec.id)
        results_path = '{}{}{}/{}'.format(cfg.paths.RESULTS, self.agent.__class__.__name__, cfg.experiment.VARIATION,
                                          env_id)
        print(plot_rewards)
        print(results_path)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        plt.plot(plot_rewards)
        print(self.env)
        plt.savefig('{}/averageRewards.png'.format(results_path))


class LifeController(SingleAgentController):
    def __init__(self, env, agent):
        super(LifeController, self).__init__(env, agent)

    def make_agents(self):
        return
