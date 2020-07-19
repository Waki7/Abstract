import logging
from typing import List, Iterable, Dict

import numpy as np
from gym import spaces

import grid_world.env_objects.core_env_objects as core_objects
import grid_world.env_objects.env_agents as core_agents
import grid_world.env_objects.landmarks as core_landmarks
import grid_world.rendering.observation_rendering as rendering
import utils.model_utils as model_utils


class CoreWorld():
    def __init__(self, cfg):
        '''
        This environment is a continuous task (non episodic)
        '''
        self.cfg = cfg

        # ---------------------------------------------------------------------------
        # set parameters from config
        # ---------------------------------------------------------------------------
        self.bounds = cfg.get('bounds', [[-1.0, 1.0], [-1.0, 1.0]])
        if not isinstance(self.bounds[0], Iterable):
            logging.info('bounds were provided in one dimension, will infer a box was desired')
            self.bounds = (self.bounds, self.bounds)
        self.dt = cfg.get('dt', .1)  # time grandularity
        observation_cfg = cfg.get('observations')

        # ---------------------------------------------------------------------------
        # initializations
        # ---------------------------------------------------------------------------
        self.object_coordinates = []
        self.landmark_map: Dict[str, core_landmarks.Landmark] = {}
        self.agent_map: Dict[str, core_agents.EnvAgent] = {}
        self.renderer = rendering.ObservationRenderer(cfg=observation_cfg)

    def get_world_obs_space(self) -> spaces.Tuple:
        '''
        this is the observation space for the world per agent, so not the global grid view, but as far how the
        world is seen by the agent, this will give the space for that, it won't include any additional data, such as
        the agent's action or communication, etc.
        :return:
        '''
        obs_spaces: List[spaces.Space] = []
        obs_spaces.append(self.renderer.get_obs_shape())
        logging.info('total of {} observable discrete observations'.format(obs_spaces[-1].high.shape))
        obs_space = spaces.Tuple(obs_spaces)
        return obs_space

    def get_world_action_space(self) -> spaces.Box:
        high = 1.0
        low = -1.0
        return spaces.Box(high=high, low=low, shape=(2,))  # x and y direction

    def reset_world(self):
        self.renderer.reset_drawing()

    def add_agent(self):
        pass

    def spawn_agents(self, agents: List[core_agents.EnvAgent], locations: List[Iterable[int]]):
        [self.spawn_agent(agent, location) for agent, location in zip(agents, locations)]

    def spawn_agent(self, agent: core_agents.EnvAgent, location: Iterable[int]):
        self.agent_map[agent.id] = agent
        agent.place(location)

    def spawn_landmarks(self, landmarks: List[core_landmarks.Landmark], locations: List[Iterable[int]]):
        [self.spawn_landmark(landmark, location) for landmark, location in zip(landmarks, locations)]

    def spawn_landmark(self, landmark: core_landmarks.Landmark, location: Iterable[int]):
        self.landmark_map[landmark.id] = landmark
        landmark.place(location)

    def is_out_of_bounds(self, location: Iterable[float]) -> bool:
        return not (self.bounds[0][0] < location[0] < self.bounds[0][1]
                    and self.bounds[1][0] < location[1] < self.bounds[1][1])

    def is_object_of_bounds(self, object: core_objects.GridObject) -> bool:
        return self.is_out_of_bounds(object.location)

    def is_legal_move(self, destination, agent=None):
        '''

        :param destination: point to see if it can be occupied
        :param agent: optionally include agent, perhaps some moves are only illegal for some agents
        :return: boolean, true or false
        '''
        pass

    def move_agent(self, agent: core_agents.EnvAgent, action: np.ndarray, illegal_func=None):
        assert agent.id in self.agent_map, 'agent has not been spawned in world'
        agent = self.agent_map[agent.id]
        location = agent.get_location()
        new_location = location + (self.dt * action)
        out_of_bounds = self.is_out_of_bounds(new_location)
        # todo logic for bouncing off and avoiding collisions, add to vector
        if not out_of_bounds:
            destination = agent.place(new_location)
            return destination
        return location

    def step_agents(self):
        pass

    def step_friendlies(self):
        pass

    def step_enemies(self):
        pass

    def get_distance(self, obj1: core_objects.GridObject, obj2: core_objects.GridObject):
        loc1, loc2 = obj1.location, obj2.location
        return model_utils.get_euclidean_distance(loc1, loc2)

    def render_world(self):
        self.renderer.reset_drawing()
        for agent in self.agent_map.values():
            pixel_location = self.renderer.convert_location_to_pixels(location=agent.location,
                                                                      origin_bounds=self.bounds)
            # self.rendererconvert_distance_to_pixels()
            self.renderer.draw_shape(agent.shape, center=pixel_location)
        for landmark in self.landmark_map.values():
            pixel_location = self.renderer.convert_location_to_pixels(location=landmark.location,
                                                                      origin_bounds=self.bounds)
            self.renderer.draw_shape(agent.shape, center=pixel_location)
        return self.renderer.get_drawing()

    def get_random_point(self):
        granularity = 1000.
        # low is inclusive, high is not, and we don't permit 1.0 to be in bounds
        rand_y = np.random.randint(low=(self.bounds[0][0] * granularity) + 1., high=self.bounds[0][1] * granularity)
        rand_x = np.random.randint(low=(self.bounds[1][0] * granularity) + 1., high=self.bounds[1][1] * granularity)
        rand_y = rand_y / granularity
        rand_x = rand_x / granularity
        return np.asarray([rand_y, rand_x])

    def initialize_empty_map(self):
        return dict(zip(self.agent_state_channels, [[], [], [], []]))

    def get_current_state(self):
        return self.state

    def get_done(self, agent):
        pass

    def get_agent_pov(self, agent: core_agents.EnvAgent):
        assert agent.id in self.agent_map, 'agent has not been spawned in world'
        pixel_location = self.renderer.convert_location_to_pixels(location=agent.location, origin_bounds=self.bounds)
        return self.renderer.get_egocentric_observation(pixel_location)

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
