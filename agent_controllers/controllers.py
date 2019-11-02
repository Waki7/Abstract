from agent_controllers.factory import register_controller
import logging

from networks.base_networks import *
from agent_algorithms.factory import ALGORITHM_REGISTRY
from networks.factory import NETWORK_REGISTERY

from shutil import copy as copy_file

import numpy as np
import os
import random
import torch
import config as cfg
from utils.storage_utils import ExperimentLogger


@register_controller
class BaseController(): # currently implemented as (i)AC
    def __init__(self, env_cfg, cfg):
        self.cfg = cfg
        self.n_agents = cfg.get(env_cfg, 1)
        self.env = gym.make(env_cfg['name'])
        self.agents = self.make_agents()
        self.sample_state = self.env.observation_space.sample()
        if isinstance(self.sample_state)

        self.experiment_logger = ExperimentLogger()
        self.is_episodic = not hasattr(self.env, 'is_episodic') or (
                hasattr(self.env, 'is_episodic') and self.env.is_episodic)



    def make_agents(self):
        n_actions = self.env.action_space.n
        n_features = self.env.observation_space.n
        critic_estimates = 1  # value estimator
        agents = []
        for i in range(0, len(self.n_agents)):
            ac_name = cfg.get('ac_name', None)
            if ac_name is not None:
                actor_network, critic_network = NETWORK_REGISTERY[ac_name](n_features,
                                                                           n_actions,
                                                                           critic_estimates,
                                                                           self.cfg['ac'],
                                                                           self.is_episodic)
            else:
                actor_network = NETWORK_REGISTERY[self.cfg['actor_name']](n_features,
                                                                             n_actions,
                                                                             self.cfg['actor'])
                critic_network = NETWORK_REGISTERY[self.cfg['critic_name']](n_features,
                                                                               critic_estimates,
                                                                               self.cfg['critic'])
            agent = ALGORITHM_REGISTRY[self.cfg['name']](self.env, actor_network, critic_network, cfg)
            agents.append(agent)

        return self.cfg['agents']

    def teach_agents(self, training_cfg):
        update_rate = 1 if self.is_episodic else cfg.experiment.UPDATE_RATE
        max_episodes = cfg.experiment.MAX_EPISODES if self.is_episodic else 1
        all_rewards = []
        numsteps = []
        avg_numsteps = []

        for episode in range(max_episodes):
            state = self.env.reset()
            rewards = []
            step = 0
            while True:
                self.step_agents(state)
                action = self.agent.step(state)
                state, reward, episode_end, _ = self.env.step(action)
                rewards.append(reward)
                self.update_agents()
                self.agent.update_policy(reward, episode_end)

                if (self.is_episodic and episode_end) or (not self.is_episodic and step % update_rate == 0):
                    numsteps.append(step)
                    avg_numsteps.append(np.mean(numsteps[-cfg.experiment.EVAL_REWARDS_WINDOW:]))
                    all_rewards.append(np.sum(rewards))
                    logging.debug(
                        "episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(
                            np.sum(rewards), decimals=3), np.round(
                            np.mean(all_rewards[-cfg.experiment.EVAL_REWARDS_WINDOW:]), decimals=3), step))
                    if self.is_episodic:
                        break
                    else:
                        rewards = []
                if step > training_cfg['timeout']:
                    break
                step += 1

        plot_rewards = [np.mean(all_rewards[i:i + 50]) for i in range(0, len(all_rewards), 50)]
        env_id = (self.env.unwrapped.spec.id if not hasattr(self.env, 'envs') else self.env.envs[0].unwrapped.spec.id)
        results_path = '{}{}{}/{}'.format(cfg.paths.RESULTS, self.agent.__class__.__name__, cfg.experiment.VARIATION,
                                          env_id)
        print(plot_rewards)
        print(results_path)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        plt.plot(plot_rewards)
        print(self.env)
        plt.savefig('{}/averageRewards.png'.format(results_path))

    def store_results(self, reward):
        self.testing_rewards.insert(self.t, reward)
        if not self.t % cfg.rewards_eval_window:
            self.average_rewards.append(np.average(self.testing_rewards.getData()))

    def step_agents(self, state):
        if self.n_agents == 1:
            self.agents[0].step(state)
        else:
            for key in self.agent_keys:
                self.agents[key].step(state[key])

    def update_agents(self):
        pass

class LifeController(BaseController):
    def __init__(self, env, agent):
        super(LifeController, self).__init__(env, agent)

    def make_agents(self):
        return
