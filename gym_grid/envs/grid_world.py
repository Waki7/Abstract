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
        self.agent_keys = cfg.get('agents', ['agent_0'])
        self.agents = [core.Agent(id=agent, observed_value=.1) for agent in self.agent_keys]

        # ---------------------------------------------------------------------------
        # initializing agents according to arbitrary naming scheme
        # ---------------------------------------------------------------------------
        self.world = core.CoreWorld(cfg)
        self.action_space = self.get_action_space()
        self.observation_space = self.get_obs_space()

        # ---------------------------------------------------------------------------
        # episodic initializations
        # ---------------------------------------------------------------------------
        self.agent_action_map = None
        self.t = 0

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
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError

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
