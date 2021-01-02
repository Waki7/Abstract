import logging
import typing as typ

import gym
import numpy as np

import envs.grid_world.env_core as core
import envs.grid_world.env_core.core_env_objects as env_objects
import envs.grid_world.env_core.env_agents as env_agents
import envs.grid_world.rendering.shapes as render_shapes
import utils.image_utils as image_utils
from envs.grid_world.env_core.action_mapper import ActionMapper
from envs.grid_world.env_variations.grid_world import GridEnv


class ManeuverSimple(GridEnv):
    def __init__(self, cfg):
        '''
        This environment is an episodic task with discrete actions and a single agent
        '''

        # will initialize world, observation_space, and action_space
        super().__init__(cfg)

        self.is_discrete_actions = True
        world_action_space = self.world.get_world_action_space()
        self.action_mapper = ActionMapper(in_space=self.action_space,
                                          out_space=world_action_space,
                                          encode_func=
                                          env_objects.get_action_unit_vector)
        self.cfg = cfg

        # ---------------------------------------------------------------------------
        # set parameters from config
        # ---------------------------------------------------------------------------
        agent_shape = render_shapes.Circle(radius=4., observed_value=55)
        self.agents = [env_agents.EnvAgent(id=agent, observed_shape=agent_shape)
                       for agent in self.agent_keys]
        self.n_agents = cfg.get('n_agents', 1)
        self.n_landmarks = cfg.get('n_landmarks', 2)
        self.n_foreign_friendlies = cfg.get('n_foreign_friendlies', 0)
        self.n_foreign_enemies = cfg.get('n_foreign_enemies', 0)
        self.agent_fov = cfg.get('agent_fov', 0.15)
        self.animation_resolution = cfg.get('animation_resolution', (100, 100))
        self.render_interpolation = cfg.get('render_interpolation')

        # ---------------------------------------------------------------------------
        # initializations
        # ---------------------------------------------------------------------------
        self.object_coordinates = []
        landmark_shape = render_shapes.Diamond(apothem=5., observed_value=200)
        self.target = env_objects.GridObject(id='target',
                                             observed_shape=landmark_shape)
        self.avoid = env_objects.GridObject(id='obstacle',
                                            observed_shape=landmark_shape)

        # ---------------------------------------------------------------------------
        # episodic initializations
        # ---------------------------------------------------------------------------
        self.global_render_frames = []
        self.agent_render_frames = dict(
            zip(self.agent_keys, [[]] * self.n_agents))
        self.agent_dones_map = dict(
            zip(self.agent_keys, [False] * self.n_agents))
        self.agent_action_map = None
        self.t = 0

    def reset(self):
        # --- get spawning locations
        landmark_locations = [self.world.get_random_point(),
                              self.world.get_random_point()]
        agent_locations = [self.world.get_random_point() for agent in
                           self.agents]
        # --- spawn the landmarks in the world, this includes placing them in the world
        self.world.spawn_landmarks([self.target, self.avoid],
                                   landmark_locations)
        self.world.spawn_agents(self.agents, agent_locations)

        # --- episodic initializations
        self.global_render_frames = []
        self.agent_render_frames = dict(
            zip(self.agent_keys, [[]] * self.n_agents))
        self.agent_dones_map = dict(
            zip(self.agent_keys, [False] * self.n_agents))
        self.agent_action_map = None
        self.t = 0

        frame = self.world.render_world()
        self.global_render_frames.append(frame)
        agent_obss = self.calc_agent_obs()
        return agent_obss

    def get_object_coordinates(self):
        # order is consistent with spawning order
        return self.world.get_object_coordinates()

    def calc_observation_space(self):
        obs_spaces = []
        world_view_spaces = self.world.get_world_obs_space().spaces
        obs_spaces.extend(world_view_spaces)
        return gym.spaces.Tuple(obs_spaces)

    def calc_action_space(self):
        action_spaces: typ.List[gym.spaces.Space] = []
        action_spaces.append(
            gym.spaces.Discrete(len(core.core_objects.DISCRETE_ACTIONS)))
        logging.info(
            'total of {} actions available'.format(action_spaces[-1].n))
        action_space = gym.spaces.Tuple(action_spaces)
        return action_space

    def add_agent(self):
        pass

    def step(self, actions: typ.Union[typ.Dict, np.long]):
        """
        Args:
            actions (Dict or np.long): an action done by the agent, encoded into its channel,
            can be a map if multi agent or one value that matches the environment's action space

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if self.n_agents == 1:
            self.world.move_agent(self.agents[0],
                                  self.action_mapper.encode(actions))
        else:
            for agent_key in actions.keys():
                action = actions[agent_key]
                # TODO REPLACE self.agents[0]
                self.world.move_agent(self.agents[0],
                                      self.action_mapper.encode(action))

        frame = self.world.render_world()
        self.global_render_frames.append(frame)
        self.t += 1

        agent_obss = self.calc_agent_obs()
        agent_rewards = self.calc_agent_rewards()
        agent_dones = self.calc_agent_dones()
        agent_infos = self.calc_agent_info()

        return agent_obss, agent_rewards, agent_dones, agent_infos

    def log_summary(self):
        logging.debug(
            '\n___________start step {}_______________'.format(self.t))

        if self.agent_action_map:
            logging.debug(
                'agent\' latest prediction : ' + str(self.agent_action_map))
        logging.debug('Env State for agent at timestep ' + str(self.t))
        for channel in self.state_map:
            val = self.state_map[channel]
            if len(val) > 0:
                logging.debug('channel {}: {}, '.format(channel, str(
                    self.state_map[channel])))
        # agent.log_predictions()
        logging.debug('env reward is : ' + str(self.current_reward))
        for object_id in self.objects.keys():
            self.objects[object_id].log_summary()
        logging.debug('___________end step {}_______________\n'.format(self.t))

    def calc_agent_obs(self):
        obs_map = {}
        for agent in self.agents:
            features = []
            map_obs = self.world.get_agent_pov(agent)
            features.append(map_obs)
            obs_map[agent.id] = features
            self.agent_render_frames[agent.id].append(map_obs)

        if self.n_agents == 1:
            return obs_map[self.agent_keys[0]]
        return obs_map

    def calc_agent_rewards(self):
        reward_map = {}
        for agent in self.agents:
            reward = -.1
            dist = self.world.get_distance(agent, self.target)
            out_of_bounds = self.world.is_object_of_bounds(agent)
            if out_of_bounds:
                reward -= 20.0
            if dist < self.agent_fov:
                reward += 1.0
            reward_map[agent.id] = reward

        if self.n_agents == 1:
            return reward_map[self.agent_keys[0]]
        return reward_map

    def calc_agent_dones(self):
        done_map = {}
        timed_out = self.t > self.timeout
        for agent in self.agents:
            prev_done = self.agent_dones_map.get(agent.id, False)
            dist = self.world.get_distance(agent, self.target)
            out_of_bounds = self.world.is_object_of_bounds(agent)
            if out_of_bounds:
                logging.error('agent was at location {}'.format(agent.location))
            done = dist < self.agent_fov or out_of_bounds or prev_done or timed_out
            done_map[agent.id] = done

        # update dones
        self.agent_dones_map = done_map

        if self.n_agents == 1:
            return done_map[self.agent_keys[0]]
        return done_map

    def calc_agent_info(self):
        info_map = {}
        return info_map

    def render(self):
        raw_frames = self.global_render_frames
        return [image_utils.convert_to_rgb_format(frame,
                                                  target_resolution=self.animation_resolution,
                                                  interpolation=self.render_interpolation)
                for frame in raw_frames]

    def render_agent_pov(self, agent_key):
        raw_frames = self.agent_render_frames[agent_key]
        return [image_utils.convert_to_rgb_format(frame,
                                                  target_resolution=self.animation_resolution,
                                                  interpolation=self.render_interpolation)
                for frame in raw_frames]
