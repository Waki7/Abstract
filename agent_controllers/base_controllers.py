import logging
from typing import Union, Dict, List

import gym
import numpy as np
import torch

import grid_world.envs as grid_env
import settings
import utils.model_utils as model_utils
from utils.env_wrappers import SubprocVecEnv
from utils.storage_utils import ExperimentLogger


def get_env_func(env_name, env_cfg):
    if len(env_cfg) > 1:
        return gym.make(env_name, cfg=env_cfg)
    else:
        return gym.make(env_cfg)


class BaseController:  # currently implemented as (i)AC
    # THIS CONTROLLER IS PASSING IN TORCH TENSORS TO THE AGENTS, AND THE ENVIRONMENTS WILL GET NUMPY ARRAYS
    def __init__(self, env_cfg, cfg):
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.env_name = env_cfg['name']
        self.env = self.make_env()

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.log_freq = cfg.get('log_freq', 50)
        self.agent_name = cfg['agent_name']

        self.agent_keys = self.env.agent_keys if hasattr(self.env, 'agent_keys') else None
        self.n_agents = 1 if self.agent_keys is None else len(self.agent_keys)

        ##########################################################################################
        # set up experiment
        ##########################################################################################
        self.is_episodic = not hasattr(self.env, 'is_episodic') or (
                hasattr(self.env, 'is_episodic') and self.env.is_episodic)
        logging.info('environment is {}'.format('episodic' if self.is_episodic else 'not episodic (continuous)'))
        self.sample_state = self.env.observation_space.sample()
        self.agents = self.make_agents()
        self.experiment_logger = ExperimentLogger()

    def make_agents(self):
        raise NotImplementedError

    def reload_env(self, env_cfg):
        self.env_cfg = env_cfg
        self.env_name = env_cfg['name']

    def make_env(self):
        if len(self.env_cfg) > 1:
            return gym.make(self.env_name, cfg=self.env_cfg)
        else:
            return gym.make(self.env_name)

    def teach_agents(self, training_cfg, experiment_folder=''):
        training = experiment_folder == ''

        n_episodes = training_cfg['n_episodes']
        checkpoint_freq = training_cfg.get('checkpoint_freq', 50)
        n_threads = self.cfg.get('n_threads', 1)
        is_batch_env = n_threads > 1

        self.experiment_logger.create_experiment(self.agent_name,
                                                 self.env_cfg['name'],
                                                 training_cfg,
                                                 experiment_folder,
                                                 env_cfg=self.env_cfg,
                                                 agent_cfg=self.cfg,
                                                 )  # this is a wraapper over summarywriter()
        env_name = self.env_name
        env_cfg = self.env_cfg
        env = SubprocVecEnv([lambda: get_env_func(env_name=env_name, env_cfg=env_cfg) for i in
                             range(n_threads)]) if is_batch_env else self.env

        step = 0
        states = env.reset()
        for episode in range(n_episodes):
            episode_lengths = [-1] * n_threads

            while True:
                actions = self.step_agents(states, is_batch_env)
                states, rewards, episode_ends, info = self.step_env(env, actions, is_batch_env)

                # self.env.log_summary()
                losses = self.update_agents(rewards, episode_ends, states)
                assert isinstance(losses, dict), 'expect losses to be returned as a dictionary'
                updated = len(losses) != 0

                self.experiment_logger.add_scalar_dict('batch_losses', losses, log=True)
                self.experiment_logger.add_agent_scalars('batch_reward', rewards, track_mean=True, track_sum=True,
                                                         log=True)

                step += 1
                for batch_idx, end in enumerate(episode_ends):
                    if episode_lengths[batch_idx] <= 0 and end:
                        episode_lengths[batch_idx] = step

                if (self.is_episodic and all(episode_ends)) or (not self.is_episodic and updated):
                    self.experiment_logger.log_progress(episode, np.mean(episode_lengths))
                    break

            self.experiment_logger.checkpoint(episode, checkpoint_freq,
                                              agents=self.agents, environment=env,
                                              render_agent_povs=isinstance(self.env, grid_env.GridEnv))
            # only reset the step if the environment is episodic
            if self.is_episodic:
                self.experiment_logger.add_agent_scalars('batch_episode_length', data=np.mean(episode_lengths),
                                                         step=episode, log=True)
                step = 0
                states = env.reset()

    def convert_obs_for_agent(self, obs, is_batch_env):
        if not is_batch_env:  # agents will always expect a batch dimension, so make batch of one
            obs = [obs, ]
        if self.n_agents == 1:
            batched_obs = model_utils.batch_env_observations(obs, self.env.observation_space)
            batched_obs = model_utils.list_to_torch_device(batched_obs)
            return batched_obs
        else:
            raise NotImplementedError('NEED TO UPDATE ENVIRONMENT OBSERVATION SPACES TO HAVE DICT FOR MULTIAGENT')
            # obs_map = {}
            # for agent in self.agents:
            #     agent_obs = model_utils.batch_env_observations(obs[agent.id])
            #     agent_obs = model_utils.list_to_torch_device(agent_obs)
            #     obs_map[agent.id] = obs.get(agent.id, None)

    def convert_env_feedback_for_agent(self, vector: Union[List[float], Dict], is_batch_env):
        if not is_batch_env:
            vector = [vector, ]
        if self.n_agents == 1:
            return torch.tensor(vector).to(settings.DEVICE)
        else:
            raise NotImplementedError('NEED TO UPDATE ENVIRONMENT OBSERVATION SPACES TO HAVE DICT FOR MULTIAGENT')

    def step_env(self, env, actions, is_batch_env):
        states, rewards, episode_ends, info = env.step(actions)
        if self.n_agents > 1:
            pass
        if not is_batch_env:
            return [states], [rewards], [episode_ends], [info]
        return states, rewards, episode_ends, info

    def step_agents(self, obs: Union[List[np.ndarray], Dict], is_batch_env):
        batched_obs = self.convert_obs_for_agent(obs, is_batch_env)
        if self.n_agents == 1:
            agent = self.agents[0]
            return agent.step(batched_obs)
        else:
            raise NotImplementedError('NEED TO UPDATE ENVIRONMENT OBSERVATION SPACES TO HAVE DICT FOR MULTIAGENT')

    def update_agents(self, reward: Union[List[float], Dict],
                      episode_end: Union[List[bool], Dict],
                      is_batch_env):
        batch_reward = self.convert_env_feedback_for_agent(reward, is_batch_env)
        batch_end = self.convert_env_feedback_for_agent(episode_end, is_batch_env)
        if self.n_agents == 1:
            loss = self.agents[0].update_policy(batch_reward, batch_end)
        else:
            raise NotImplementedError('NEED TO UPDATE ENVIRONMENT OBSERVATION SPACES FOR THE NEW_STATE')
            # agent_reward = []
            # agent_ends = []
            # agent_state = []
            #
            # loss = {}
            # for key in self.agent_keys:
            #     loss[key] = (self.agents[key].update_policy(
            #         reward[key], episode_end[key], new_state[key]
            #     ))
        return loss
