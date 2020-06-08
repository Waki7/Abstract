import logging

import grid_world.env_objects as core
import grid_world.envs.grid_world as grid_world


class ManeuverSimple(grid_world.GridEnv):
    def __init__(self, cfg):
        '''
        This environment is a continuous task (non episodic)
        '''
        super().__init__(cfg)
        self.cfg = cfg

        # ---------------------------------------------------------------------------
        # set parameters from config
        # ---------------------------------------------------------------------------
        self.n_agents = cfg.get('n_agents', 1)
        self.n_landmarks = cfg.get('n_landmarks', 10)
        self.n_foreign_friendlies = cfg.get('foreign_friendlies', [])
        self.n_foreign_enemies = cfg.get('foreign_enemies', [])

        # ---------------------------------------------------------------------------
        # initializations
        # ---------------------------------------------------------------------------
        self.object_coordinates = []
        self.target = core.GridObject(id='target', observed_value=.4)
        self.avoid = core.GridObject(id='obstacle', observed_value=.8)

        # ---------------------------------------------------------------------------
        # episodic initializations
        # ---------------------------------------------------------------------------
        self.agent_action_map = None
        self.t = 0
        self.reset()

    def reset(self):
        # --- get spawning locations
        landmark_locations = [self.world.get_random_point(), self.world.get_random_point()]
        agent_locations = [self.world.get_random_point() for agent in self.agents]
        # --- spawn the landmarks in the world, this includes placing them in the world
        self.world.spawn_landmarks([self.target, self.avoid], landmark_locations)
        self.world.spawn_agents(self.agents, agent_locations)

        agent_obss = self.calc_agent_obs()
        return agent_obss

    def get_obs_space(self):
        return self.world.get_obs_space()

    def get_action_space(self):
        return self.world.get_action_space()

    def add_agent(self):
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
        for agent in agent_actions.keys():
            action = agent_actions[agent]
            self.world.move_agent(agent, action)

        self.world.render_world()

        agent_obss = self.calc_agent_obs()
        agent_rewards = self.calc_agent_rewards()
        agent_dones = self.calc_agent_dones()
        agent_infos = self.calc_agent_info()

        return agent_obss, agent_rewards, agent_dones, agent_infos

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

    def calc_agent_obs(self):
        obs_map = {}
        for agent in self.agents:
            obs_map[agent.id] = self.world.get_agent_pov(agent)

        if self.n_agents == 1:
            return obs_map[self.agent_keys[0]]
        return obs_map

    def calc_agent_rewards(self):
        pass

    def calc_agent_info(self):
        pass

    def calc_agent_info(self):
        pass
