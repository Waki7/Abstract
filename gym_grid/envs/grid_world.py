import logging

import gym

import gym_grid.env_objects as core


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
        self.n_agents = cfg.get('n_agents', 1)
        self.n_landmarks = cfg.get('n_landmarks', 10)
        self.foreign_friendlies = cfg.get('foreign_friendlies', [])
        self.foreign_enemies = cfg.get('foreign_enemies', [])

        # ---------------------------------------------------------------------------
        # initializing agents according to arbitrary naming scheme
        # ---------------------------------------------------------------------------
        self.world = core.CoreWorld(cfg)
        self.action_space = self.get_action_space()
        self.observation_space = self.get_obs_space()

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
        raise NotImplementedError

    def load_agents(self):
        '''
        When rerunning agents, this method will allow you to update any shapes to fit the parameters
        todo
        :return:
        '''
        pass

    def render(self):
        raise NotImplementedError

    def get_obs_space(self):
        return self.world.get_obs_space()

    def get_action_space(self):
        return self.world.get_action_space()

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
