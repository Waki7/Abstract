# import logging
# from typing import Union, Dict, List
#
# import numpy as np
# from gym import spaces
#
# from envs import grid_world as core, grid_world as grid_world
#
#
# class ManeuverMulti(grid_world.GridEnv):
#     def __init__(self, cfg):
#         '''
#         This environment is an episodic task with discrete actions and a single agent
#         '''
#
#         # will initialize world, observation_space, and action_space
#         super().__init__(cfg)
#
#         self.is_discrete_actions = True
#         world_action_space = self.world.get_world_action_space()
#         self.action_mapper = core.ActionMapper(in_space=self.action_space,
#                                                out_space=world_action_space,
#                                                encode_func=core.get_action_unit_vector)
#         self.cfg = cfg
#
#         # ---------------------------------------------------------------------------
#         # set parameters from config
#         # ---------------------------------------------------------------------------
#         self.n_agents = cfg.get('n_agents', 1)
#         self.n_landmarks = cfg.get('n_landmarks', 10)
#         self.n_foreign_friendlies = cfg.get('foreign_friendlies', [])
#         self.n_foreign_enemies = cfg.get('foreign_enemies', [])
#         self.agent_fov = cfg.get('agent_fov', 0.1)
#
#         # ---------------------------------------------------------------------------
#         # initializations
#         # ---------------------------------------------------------------------------
#         self.object_coordinates = []
#         self.target = core.GridObject(id='target', observed_value=.4)
#         self.avoid = core.GridObject(id='obstacle', observed_value=.8)
#
#         # ---------------------------------------------------------------------------
#         # episodic initializations
#         # ---------------------------------------------------------------------------
#         self.agent_action_map = None
#         self.t = 0
#         self.reset()
#
#     def reset(self):
#         # --- get spawning locations
#         landmark_locations = [self.world.get_random_point(), self.world.get_random_point()]
#         agent_locations = [self.world.get_random_point() for agent in self.agents]
#         # --- spawn the landmarks in the world, this includes placing them in the world
#         self.world.spawn_landmarks([self.target, self.avoid], landmark_locations)
#         self.world.spawn_agents(self.agents, agent_locations)
#
#         agent_obss = self.calc_agent_obs()
#         return agent_obss
#
#     def calc_observation_space(self):
#         obs_spaces = []
#         world_view_spaces = self.world.get_world_obs_space().spaces
#         obs_spaces.extend(world_view_spaces)
#         return spaces.Tuple(obs_spaces)
#
#     def calc_action_space(self):
#         action_spaces: List[spaces.Space] = []
#         action_spaces.append(spaces.Discrete(len(core.core_objects.DISCRETE_ACTIONS)))
#         logging.info('total of {} actions available'.format(action_spaces[-1].n))
#         action_space = spaces.Tuple(action_spaces)
#         return action_space
#
#     def add_agent(self):
#         pass
#
#     def step(self, agent_actions: Union[Dict, np.long]):
#         """
#         Args:
#             agent_actions (object): an action done by the agent, encoded into its channel
#
#         Returns:
#             observation (object): agent's observation of the current environment
#             reward (float) : amount of reward returned after previous action
#             done (bool): whether the episode has ended, in which case further step() calls will return undefined results
#             info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
#         """
#         if self.n_agents == 1:
#             self.world.move_agent(self.agents[0],
#                                   self.action_mapper.encode(agent_actions))
#
#         for agent_key in agent_actions.keys():
#             action = agent_actions[agent_key]
#             self.world.move_agent(self.agents[0], self.action_mapper.encode(action))
#
#         self.world.render_world()
#
#         agent_obss = self.calc_agent_obs()
#         agent_rewards = self.calc_agent_rewards()
#         agent_dones = self.calc_agent_dones()
#         agent_infos = self.calc_agent_info()
#
#         return agent_obss, agent_rewards, agent_dones, agent_infos
#
#     def initialize_empty_map(self):
#         return dict(zip(self.agent_state_channels, [[], [], [], []]))
#
#     def get_current_state(self):
#         return self.state
#
#     def log_summary(self):
#         logging.debug('\n___________start step {}_______________'.format(self.t))
#
#         if self.agent_action_map:
#             logging.debug('agent\' latest prediction : ' + str(self.agent_action_map))
#         logging.debug('Env State for agent at timestep ' + str(self.t))
#         for channel in self.state_map:
#             val = self.state_map[channel]
#             if len(val) > 0:
#                 logging.debug('channel {}: {}, '.format(channel, str(self.state_map[channel])))
#         # agent.log_predictions()
#         logging.debug('env reward is : ' + str(self.current_reward))
#         for object_id in self.objects.keys():
#             self.objects[object_id].log_summary()
#         logging.debug('___________end step {}_______________\n'.format(self.t))
#
#     def calc_agent_obs(self):
#         obs_map = {}
#         for agent in self.agents:
#             features = []
#             map_obs = self.world.get_agent_pov(agent)
#             features.append(map_obs)
#             obs_map[agent.id] = features
#
#         if self.n_agents == 1:
#             return obs_map[self.agent_keys[0]]
#         return obs_map
#
#     def calc_agent_rewards(self):
#         obs_map = {}
#         for agent in self.agents:
#             self.world.get_dist(agent.ke)
#             features = []
#             map_obs = self.world.get_agent_pov(agent)
#             features.append(map_obs)
#             obs_map[agent.id] = features
#
#         if self.n_agents == 1:
#             return obs_map[self.agent_keys[0]]
#         return obs_map
#
#     def calc_agent_info(self):
#         info_map = {}
#         return info_map
