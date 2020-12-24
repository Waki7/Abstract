from typing import Union, Dict, List

import gym
import numpy as np
import torch

import settings
import utils.model_utils as model_utils
from utils.env_wrappers import SubprocVecEnv
from utils.experiment_utils import ExperimentLogger


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
        self.is_episodic = False
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
        self.sample_state = self.env.observation_space.sample()
        self.agents = self.make_agents()
        self.agent_map = dict(zip(self.agent_keys, self.agents))
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
        for episode in range(n_episodes):
            step = 0
            states = self.reset_env(env, is_batch_env)

            episode_lengths = [-1] * n_threads
            while True:
                actions = self.step_agents(states, is_batch_env)
                states, rewards, episode_ends, info = self.step_env(env, actions, is_batch_env)

                # self.env.log_summary()
                losses = self.update_agents(rewards, episode_ends, states)
                assert isinstance(losses, dict), 'expect losses to be returned as a dictionary'

                self.experiment_logger.add_scalar_dict('batch_losses', losses, log=True)
                self.experiment_logger.add_agent_scalars('batch_reward', rewards, track_mean=True, track_sum=True,
                                                         log=True)

                step += 1
                for batch_idx, end in enumerate(episode_ends):
                    if episode_lengths[batch_idx] <= 0 and end:
                        episode_lengths[batch_idx] = step

                if all(episode_ends):
                    self.experiment_logger.log_progress(episode, np.mean(episode_lengths))
                    break

            self.experiment_logger.checkpoint(episode, checkpoint_freq,
                                              agent_map=self.agent_map, environment=env)
            # only reset the step if the environment is episodic
            self.experiment_logger.add_agent_scalars('batch_episode_length', data=np.mean(episode_lengths),
                                                     step=episode, log=True)

    def convert_obs_for_agent(self, obs: Union[List[np.ndarray], Dict]):
        if self.n_agents == 1:
            batched_obs = model_utils.batch_env_observations(obs, self.env.observation_space)
            batched_obs = model_utils.scale_space(state=batched_obs, space=self.env.observation_space)
            batched_obs = model_utils.nd_list_to_torch(batched_obs)
            return batched_obs
        else:
            raise NotImplementedError('NEED TO UPDATE ENVIRONMENT OBSERVATION SPACES TO HAVE DICT FOR MULTIAGENT')
            # obs_map = {}
            # for agent in self.agents:
            #     agent_obs = model_utils.batch_env_observations(obs[agent.id])
            #     agent_obs = model_utils.list_to_torch_device(agent_obs)
            #     obs_map[agent.id] = obs.get(agent.id, None)

    def convert_env_feedback_for_agent(self, rewards: Union[List[float], Dict],
                                       episode_ends: Union[List[float], Dict], is_batch_env):
        if not is_batch_env:
            rewards = [rewards, ]
            episode_ends = [episode_ends, ]
        if self.n_agents == 1:
            return torch.tensor(rewards).to(**settings.ARGS), torch.tensor(episode_ends).to(settings.DEVICE)
        else:
            raise NotImplementedError('NEED TO UPDATE ENVIRONMENT OBSERVATION SPACES TO HAVE DICT FOR MULTIAGENT')

    def reset_env(self, env, is_batch_env: Union[SubprocVecEnv, gym.Env]):
        if not is_batch_env:
            return [env.reset()]
        return env.reset()

    def step_env(self, env, actions, is_batch_env):
        states, rewards, episode_ends, info = env.step(actions)
        if self.n_agents > 1:
            pass
        if not is_batch_env:
            return [states], [rewards], [episode_ends], [info]
        return states, rewards, episode_ends, info

    def step_agent(self, agent, batched_obs):
        actions = agent.step(batched_obs)
        return actions

    def step_agents(self, obs: Union[List[np.ndarray], Dict], is_batch_env):
        batched_obs = self.convert_obs_for_agent(obs)
        if self.n_agents == 1:
            agent = self.agents[0]
            actions = self.step_agent(agent, batched_obs)
            if not is_batch_env:
                actions = actions[0]  # TODO, MAYBE MOVE THIS TO AGENT SIDE, WHO SHOULD DECIDE IF BATCH DIM OR NOT
        else:
            raise NotImplementedError('NEED TO UPDATE ENVIRONMENT OBSERVATION SPACES TO HAVE DICT FOR MULTIAGENT')
        return actions

    def update_agents(self, rewards: Union[List[float], Dict],
                      episode_ends: Union[List[bool], Dict],
                      is_batch_env):
        batch_reward, batch_end = self.convert_env_feedback_for_agent(rewards=rewards,
                                                                      episode_ends=episode_ends,
                                                                      is_batch_env=is_batch_env)
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
