import logging
from typing import List, Iterable

import numpy as np
from gym import spaces

import grid_world.env_objects.core_env_objects as core_objects
import grid_world.env_objects.env_agents as core_agents
import grid_world.env_objects.landmarks as core_landmarks
import grid_world.rendering.observation_rendering as rendering


class CoreWorld():
    def __init__(self, cfg):
        '''
        This environment is a continuous task (non episodic)
        '''
        self.cfg = cfg

        # ---------------------------------------------------------------------------
        # set parameters from config
        # ---------------------------------------------------------------------------
        self.bounds = cfg.get('bounds', [-1.0, 1.0])
        self.dt = cfg.get('dt', .1)  # time grandularity
        observation_cfg = cfg.get('observations')

        # ---------------------------------------------------------------------------
        # initializations
        # ---------------------------------------------------------------------------
        self.object_coordinates = []
        self.landmark_map = {}
        self.agent_map = {}
        self.renderer = rendering.ObservationRenderer(cfg=observation_cfg)

    def get_obs_space(self):
        obs_spaces: List[spaces.Space] = []
        obs_spaces.append(self.renderer.get_obs_shape())
        logging.info('total of {} observable discrete observations'.format(obs_spaces[-1].high.shape))
        obs_space = spaces.Tuple(obs_spaces)
        return obs_space

    def get_action_space(self):
        action_spaces: List[spaces.Space] = []
        action_spaces.append(spaces.Discrete(len(core_objects.ACTIONS)))
        logging.info('total of {} actions available'.format(action_spaces[-1].n))
        action_space = spaces.Tuple(action_spaces)
        return action_space

    def reset_world(self):
        self.renderer.reset_drawing()

    def add_agent(self):
        pass

    def spawn_agents(self, agents: List[core_agents.Agent], locations: List[Iterable[int]]):
        [self.spawn_agent(agent, location) for agent, location in zip(agents, locations)]

    def spawn_agent(self, agent: core_agents.Agent, location: Iterable[int]):
        self.agent_map[agent.id] = agent
        agent.place(location)

    def spawn_landmarks(self, landmarks: List[core_landmarks.Landmark], locations: List[Iterable[int]]):
        [self.spawn_landmark(landmark, location) for landmark, location in zip(landmarks, locations)]

    def spawn_landmark(self, landmark: core_landmarks.Landmark, location: Iterable[int]):
        self.landmark_map[landmark.id] = landmark
        landmark.place(location)

    def is_legal_move(self, destination, agent=None):
        '''

        :param destination: point to see if it can be occupied
        :param agent: optionally include agent, perhaps some moves are only illegal for some agents
        :return: boolean, true or false
        '''
        pass

    def move_agent(self, agent: core_agents.Agent, action: np.ndarray):
        assert agent.id in self.agent_map, 'agent has not been spawned in world'
        agent = self.agent_map[agent]
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

    def render_world(self):
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
        return np.asarray([rand_y, rand_x])

    def initialize_empty_map(self):
        return dict(zip(self.agent_state_channels, [[], [], [], []]))

    def get_current_state(self):
        return self.state

    def get_done(self, agent):
        pass

    def get_agent_pov(self, agent: core_agents.Agent):
        assert agent.id in self.agent_map, 'agent has not been spawned in world'
        return self.renderer.get_frame(agent.location)

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
