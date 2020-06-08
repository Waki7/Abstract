import gym

from utils.env_wrappers import SubprocVecEnv
from utils.storage_utils import ExperimentLogger


def get_env_func(env_name, env_cfg):
    if len(env_cfg) > 1:
        return gym.make(env_name, cfg=env_cfg)
    else:
        return gym.make(env_cfg)


class BaseController:  # currently implemented as (i)AC
    def __init__(self, env_cfg, cfg):
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.env_name = env_cfg['name']
        self.env = self.make_env()

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.log_freq = cfg.get('log_freq', 50)
        self.agent_name = cfg['agent_name']

        self.agent_keys = self.env.agent_keys if hasattr(self.env, 'agent_keys') else None
        self.n_agents = 1 if self.agent_keys is None else len(self.agent_keys)

        ##########################################################################################
        # set up experiment
        ##########################################################################################
        self.is_episodic = not hasattr(self.env, 'is_episodic') or (
                hasattr(self.env, 'is_episodic') and self.env.is_episodic)
        self.sample_state = self.env.observation_space.sample()
        self.agents = self.make_agents()
        self.experiment_logger = ExperimentLogger()

    def make_agents(self):
        raise NotImplementedError

    def reload_env(self, env_cfg):
        self.env_cfg = env_cfg
        self.env_name = env_cfg['name']

    def make_env(self):
        if len(self.env_cfg) > 1:
            return gym.make(self.env_name, cfg=self.env_cfg)
        else:
            return gym.make(self.env_name)

    def teach_agents(self, training_cfg, experiment_folder=''):
        training = experiment_folder == ''

        n_episodes = training_cfg['n_episodes']
        n_threads = self.cfg.get('n_threads', 1)
        is_batch_env = n_threads > 1

        self.experiment_logger.create_experiment(self.agent_name,
                                                 self.env_cfg['name'],
                                                 training_cfg,
                                                 experiment_folder,
                                                 env_cfg=self.env_cfg,
                                                 agent_cfg=self.cfg,
                                                 )  # this is a wraapper over summarywriter()
        step = 0
        env_name = self.env_name
        env_cfg = self.env_cfg
        env = SubprocVecEnv([lambda: get_env_func(env_name=env_name, env_cfg=env_cfg) for i in
                             range(n_threads)]) if is_batch_env else self.env

        state = env.reset()
        for episode in range(n_episodes):
            while True:
                actions = self.step_agents(state, is_batch_env)
                state, reward, episode_end, info = env.step(actions)
                # self.env.log_summary()
                losses = self.update_agents(reward, episode_end, state)
                assert isinstance(losses, dict), 'expect losses to be returned as a dictionary'
                updated = len(losses) != 0

                self.experiment_logger.add_scalar_dict('losses', losses, log=True)
                self.experiment_logger.add_agent_scalars('reward', reward, track_mean=True, track_sum=True, log=True)

                if (self.is_episodic and episode_end) or (not self.is_episodic and updated):
                    self.experiment_logger.log_progress(episode, step)
                    if self.is_episodic:
                        self.experiment_logger.add_agent_scalars('episode_length', data=step, step=episode, log=True)

                if (self.is_episodic and episode_end) or (not self.is_episodic and updated):
                    break

                step += 1

            # only reset the step if the environment is episodic
            if self.is_episodic:
                step = 0
                state = self.env.reset()

    def step_agents(self, state, is_batch_env):
        if self.n_agents == 1:
            if is_batch_env:
                raise NotImplementedError
            else:
                return self.agents[0].step(state)
        else:
            if is_batch_env:
                raise NotImplementedError
            else:
                return [self.agents[key].step(state[key]) for key in self.agent_keys]

    def update_agents(self, reward, episode_end, new_state):
        if self.n_agents == 1:
            loss = self.agents[0].update_policy(reward, episode_end, new_state)
        else:
            loss = {}
            for key in self.agent_keys:
                loss[key] = (self.agents[key].update_policy(
                    reward[key], episode_end[key], new_state[key]
                ))
        return loss
