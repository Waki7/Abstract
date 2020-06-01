import logging

import gym_grid.env_objects as core
import gym_grid.envs.grid_world as grid_world


class GridEnv(grid_world.GridEnv):
    def __init__(self, cfg):
        '''
        This environment is a continuous task (non episodic)
        '''
        self.cfg = cfg

        # ---------------------------------------------------------------------------
        # set parameters from config
        # ---------------------------------------------------------------------------
        self.agent_keys = cfg.get('agents', ['agent_0'])
        self.n_agents = cfg.get('n_agents', 1)
        self.n_landmarks = cfg.get('n_landmarks', 10)
        self.n_foreign_friendlies = cfg.get('foreign_friendlies', [])
        self.n_foreign_enemies = cfg.get('foreign_enemies', [])

        self.agents = [core.Agent(id=agent, observed_value=.1) for agent in self.agent_keys]

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
        self.target.place(self.world.get_random_point())
        self.avoid.place(self.world.get_random_point())
        self.world.spawn_landmarks([self.target, self.avoid])
        self.world.spawn_agents(self.agents)


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

        agent_obss = self.calc_agent_obs()
        agent_rewards = self.calc_agent_rewards()
        agent_dones = self.calc_agent_dones()
        agent_infos = self.calc_agent_info()

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
        self.world.draw()
        for agent in self.agents:
            obs_map[agent.id] = self.world.get_obs(agent)

    def calc_agent_rewards(self):
        pass

    def calc_agent_info(self):
        pass

    def calc_agent_info(self):
        pass
