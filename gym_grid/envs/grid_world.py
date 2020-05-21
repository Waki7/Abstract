import logging
from enum import Enum

import gym
import gym_grid.envs.grid_objects as objects
import numpy as np
import torch
from gym import spaces


class GridEnv(gym.Env):
    def __init__(self, cfg):
        '''
        This environment is a continuous task (non episodic)
        '''
        self.cfg = cfg

        # ---------------------------------------------------------------------------
        # set parameters from config
        # ---------------------------------------------------------------------------
        self.height = cfg['height']
        self.width = cfg['width']
        self.n_agents = cfg.get('n_agents', 1)
        self.n_landmarks = cfg.get('n_landmarks', 10)
        self.foreign_friendlies = cfg.get('foreign_friendlies', [])
        self.foreign_enemies = cfg.get('foreign_enemies', [])

        # ---------------------------------------------------------------------------
        # initializing agents according to arbitrary naming scheme
        # ---------------------------------------------------------------------------

        self.grid = torch.zeros((self.height, self.width))

        self.action_space = spaces.Discrete(len(objects.ACTIONS))
        high = np.zeros_like(self.grid)
        low = np.ones_like(self.grid)
        self.observation_space = spaces.Box(high=high, low=low)
        logging.info('total of {} actions available'.format(self.action_space.n))
        logging.info('total of {} observable discrete observations'.format(self.observation_space.high.shape))

        # ---------------------------------------------------------------------------
        # initializations
        # ---------------------------------------------------------------------------
        self.object_coordinates = []

        # ---------------------------------------------------------------------------
        # episodic initializations
        # ---------------------------------------------------------------------------
        self.agent_action_map = None
        self.t = 0
        self.hunger_level = 0
        self.thirst_level = 0
        self.current_reward = 0
        self.reset()

    def reset(self):
        for i in range(0, self.n_landmarks):
            y = torch.randint(high=self.height, size=(1,)).item()
            x = torch.randint(high=self.width, size=(1,)).item()
            point = (y, x)
            self.object_coordinates.append(point)

    def add_agent(self):
        pass

    def step_enum(self, agent_action_map: Enum):  # this will not be an enum for long i think
        agent_feel = []
        self.agent_action_map = agent_action_map
        # self.log_summary(agent_action_map)

        return self.state, self.current_reward, False, {}

    def step(self, agent_actions):
        """
        Args:
            action (object): an action done by the agent, encoded into its channel

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        return self.step_enum(None)

    def initialize_empty_map(self):
        return dict(zip(self.agent_state_channels, [[], [], [], []]))

    def get_current_state(self):
        return self.state

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
