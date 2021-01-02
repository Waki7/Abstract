import logging
import typing as typ

import gym
import numpy as np

from envs import grid_world as core


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
        self.agent_keys = cfg.get('agents', ['agent_0'])

        # ---------------------------------------------------------------------------
        # initializing agents according to arbitrary naming scheme
        # ---------------------------------------------------------------------------
        self.world = core.CoreWorld(cfg)
        self.action_space = self.calc_action_space()
        self.observation_space = self.calc_observation_space()

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

    def convert_action(self, action: typ.Union[int, np.ndarray]):
        '''
        fall back action conversion, u can create an action mapper for more particular behavior
        todo consider how design this vs action mapper, which is better, or both is good?
        :param action:
        :return:
        '''
        if isinstance(self.action_space, gym.spaces.Discrete):
            return core.get_action_unit_vector(action)
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            # todo zero center, assuming the action will be 0 (backward), 1 (stay), 2 (forward) for each component
            raise NotImplementedError('no implementation for MultiDiscrete')
        if isinstance(self.action_space, gym.spaces.Box):
            raise NotImplementedError('no implementation for continuous actions')
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def get_spaces(self) -> typ.Tuple[gym.Space, gym.Space]:
        return self.observation_space, self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def calc_observation_space(self) -> gym.Space:
        raise NotImplementedError

    def calc_action_space(self) -> gym.Space:
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
