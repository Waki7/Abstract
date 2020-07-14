import logging
from typing import Union

import gym
import numpy as np
from gym import spaces

import grid_world.env_objects as core


class GridEnv(gym.Env):
    def __init__(self, cfg):
        '''
        This environment is a continuous task (non episodic)
        '''
        self.cfg = cfg

        # ---------------------------------------------------------------------------
        # set parameters from config
        # ---------------------------------------------------------------------------
        self.timeout = cfg['timeout']
        self.n_agents = cfg.get('n_agents', 1)
        self.agent_keys = cfg.get('agents', ['agent_0'])
        self.agents = [core.EnvAgent(id=agent, observed_value=55) for agent in self.agent_keys]

        # ---------------------------------------------------------------------------
        # initializing agents according to arbitrary naming scheme
        # ---------------------------------------------------------------------------
        self.world = core.CoreWorld(cfg)
        self.action_space = self.get_action_space()
        self.observation_space = self.get_obs_space()

        # ---------------------------------------------------------------------------
        # episodic initializations
        # ---------------------------------------------------------------------------
        self.agent_action_map = None
        self.t = 0

    def reset(self):
        raise NotImplementedError

    def load_agents(self):
        '''
        When rerunning agents, this method will allow you to update any shapes to fit the parameters
        todo
        :return:
        '''
        pass

    def convert_action(self, action: Union[int, np.ndarray]):
        '''
        fall back action conversion, u can create an action mapper for more particular behavior
        todo consider how design this vs action mapper, which is better, or both is good?
        :param action:
        :return:
        '''
        if isinstance(self.action_space, spaces.Discrete):
            return core.get_action_unit_vector(action)
        if isinstance(self.action_space, spaces.MultiDiscrete):
            # todo zero center, assuming the action will be 0 (backward), 1 (stay), 2 (forward) for each component
            raise NotImplementedError('no implementation for MultiDiscrete')
        if isinstance(self.action_space, spaces.Box):
            raise NotImplementedError('no implementation for continuous actions')
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def get_spaces(self):
        return self.observation_space, self.action_space

    def get_obs_space(self):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError

    def log_summary(self):
        logging.debug('\n___________start step {}_______________'.format(self.t))

        if self.agent_action_map:
            logging.debug('agent\' latest prediction : ' + str(self.agent_action_map))
        logging.debug('Env State for agent at timestep ' + str(self.t))
        for channel in self.state_map:
            val = self.state_map[channel]
            if len(val) > 0:
                logging.debug('channel {}: {}, '.format(channel, str(self.state_map[channel])))
        # agent.log_predictions()
        logging.debug('env reward is : ' + str(self.current_reward))
        for object_id in self.objects.keys():
            self.objects[object_id].log_summary()
        logging.debug('___________end step {}_______________\n'.format(self.t))
