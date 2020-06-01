import logging
from typing import List

import numpy as np
import torch
from gym import spaces

import gym_grid.env_objects.core_env_objects as core_objects
import gym_grid.env_objects.env_agents as core_agents
import gym_grid.env_objects.landmarks as core_landmarks
import gym_grid.envs.grid_world as grid_world
import gym_grid.rendering.observation_rendering as rendering


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

        self.dt = cfg.get('dt', .1)  # time grandularity

        # ---------------------------------------------------------------------------
        # initializations
        # ---------------------------------------------------------------------------
        self.object_coordinates = []
        self.landmark_map = {}
        self.agent_map = {}
        self.renderer = rendering.ObservationRenderer(resolution=self.resolution)

    def get_obs_space(self):
        high = 1.
        low = 0.
        obs_space = spaces.Box(high=high, low=low, shape=(self.agent_resolution, self.agent_resolution))
        logging.info('total of {} observable discrete observations'.format(obs_space.high.shape))
        return obs_space

    def get_action_space(self):
        action_space = spaces.Discrete(len(core_objects.ACTIONS))
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

    def spawn_agents(self, agents: List[core_agents.Agent]):
        for agent in agents:
            self.agent_map[agent.id] = agent

    def spawn_agent(self, agent: core_agents.Agent):
        self.agent_map[agent.id] = agent

    def spawn_landmarks(self, landmarks: List[core_landmarks.Landmark]):
        for landmark in landmarks:
            self.landmark_map[landmark.id] = landmark

    def spawn_landmark(self, landmark: core_landmarks.Landmark):
        self.landmark_map[landmark.id] = landmark

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
        # todo logic for bouncing off and avoiding collisions, add to vector
        destination = agent.place(new_location)
        return destination

    def step_agents(self):
        pass

    def step_friendlies(self):
        pass

    def step_enemies(self):
        pass

    def draw(self):
        self.renderer.reset_drawing()
        for agent in self.agent_map.values():
            self.renderer.draw_circle(center=agent.location, radius=1.)
        for landmark in self.landmark_map.values():
            self.renderer.draw_square(center=landmark.location, length=1.)

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

    def get_done(self, agent):
        pass

    def get_obs(self, agent):
        assert self.agent_map.get(agent.id, None) is not None, 'agent has not been spawned in world'
        return self.renderer.get_obs(agent.location)

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
