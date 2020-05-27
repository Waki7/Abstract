import logging

import numpy as np
import torch
from gym import spaces

import gym_grid.env_objects as core
import gym_grid.env_objects.core_env_objects as objects
import gym_grid.envs.grid_world as grid_world
import gym_grid.rendering.environment_rendering as rendering


class CoreWorld(grid_world.GridEnv):
    def __init__(self, cfg):
        '''
        This environment is a continuous task (non episodic)
        '''
        self.cfg = cfg

        # ---------------------------------------------------------------------------
        # set parameters from config
        # ---------------------------------------------------------------------------
        self.bounds = cfg.get('bounds', [-1.0, 1.0])
        self.resolution = cfg.get('resolution', 100)
        self.agent_resolution = cfg.get('agent_resolution', self.resolution)

        self.n_agents = cfg.get('n_agents', 1)
        self.n_foreign_friendlies = cfg.get('foreign_friendlies', [])
        self.n_foreign_enemies = cfg.get('foreign_enemies', [])
        self.agent_list = cfg.get('agents', ['agent_0'])
        self.dt = cfg.get('dt', .1)  # time grandularity

        self.agent_map = [core.Agent(id=agent) for agent in self.agent_list]

        # ---------------------------------------------------------------------------
        # initializations
        # ---------------------------------------------------------------------------
        self.object_coordinates = []
        self.renderer = rendering.EnvironmentRenderer(resolution=self.resolution)

    def get_obs_space(self):
        high = 1.
        low = 0.
        obs_space = spaces.Box(high=high, low=low, shape=(self.agent_resolution, self.agent_resolution))
        logging.info('total of {} observable discrete observations'.format(obs_space.high.shape))
        return obs_space

    def get_action_space(self):
        action_space = spaces.Discrete(len(objects.ACTIONS))
        logging.info('total of {} actions available'.format(action_space.n))
        return action_space

    def reset(self):

        self.grid = torch.zeros((self.height, self.width))

        for i in range(0, self.n_landmarks):
            y = torch.randint(high=self.height, size=(1,)).item()
            x = torch.randint(high=self.width, size=(1,)).item()
            point = (y, x)
            self.object_coordinates.append(point)
        for i in range(0, self.n_foreign_friendlies):
            pass
        for i in range(0, self.n_foreign_friendlies):
            pass

    def add_agent(self):
        pass

    def is_legal_move(self, destination, agent=None):
        '''

        :param destination: point to see if it can be occupied
        :param agent: optionally include agent, perhaps some moves are only illegal for some agents
        :return: boolean, true or false
        '''
        pass

    def move_agent(self, agent_key, action: np.ndarray):
        agent = self.agent_map[agent_key]
        location = agent.get_location()
        new_location = location + (self.dt * action)
        destination = agent.place(new_location)

    def step_agents(self):
        pass

    def step_friendlies(self):
        pass

    def step_enemies(self):
        pass

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
        for agent_key in agent_actions.keys():
            self.move_agent(agent_key=agent_key, action=agent_actions[agent_key])

    def get_random_point(self):
        granularity = 1000.
        rand_x = np.random.randint(low=self.bounds[0] * granularity, high=self.bounds[1] * granularity)
        rand_y = np.random.randint(low=self.bounds[0] * granularity, high=self.bounds[1] * granularity)
        rand_x = rand_x / granularity
        rand_y = rand_y / granularity
        return np.asarray(rand_y, rand_x)

    def initialize_empty_map(self):
        return dict(zip(self.agent_state_channels, [[], [], [], []]))

    def get_current_state(self):
        return self.state

    def get_done(self, agent_key):
        pass

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
