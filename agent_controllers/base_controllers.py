from agent_controllers.factory import register_controller
from networks.base_networks import *
from agent_algorithms.factory import AGENT_REGISTRY
from networks.factory import get_network
from utils.storage_utils import ExperimentLogger


class BaseController:  # currently implemented as (i)AC
    def __init__(self, env_cfg, cfg):
        self.cfg = cfg
        self.env_cfg = env_cfg

        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.log_freq = cfg.get('log_freq', 50)
        self.agent_name = cfg['agent_name']
        if len(env_cfg) > 1:
            self.env = gym.make(env_cfg['name'], cfg=env_cfg)
        else:
            self.env = gym.make(env_cfg['name'])
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
        if isinstance(self.env.observation_space, gym.spaces.Box):
            n_features = self.env.observation_space.shape[0]
        else:
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                n_features = self.env.observation_space.n
            else:
                raise NotImplementedError
        n_actions = self.env.action_space.n
        critic_estimates = 1  # value estimator
        agents = []
        for i in range(0, self.n_agents):
            if self.ac_name is not None:

                ac_network = get_network(key=self.ac_name,
                                         out_shape=n_actions,
                                         out_shape2=critic_estimates,
                                         cfg=self.ac_cfg,
                                         n_features=n_features)
                agent = AGENT_REGISTRY[self.agent_name](self.is_episodic,
                                                        self.cfg,
                                                        ac_network)
            else:
                actor_network = get_network(key=self.actor_name,
                                            out_shape=n_actions,
                                            cfg=self.actor_cfg,
                                            n_features=n_features)
                critic_network = get_network(key=self.critic_name,
                                             out_shape=critic_estimates,
                                             cfg=self.critic_cfg,
                                             n_features=n_features)
                agent = AGENT_REGISTRY[self.agent_name](self.is_episodic,
                                                        self.cfg,
                                                        actor_network,
                                                        critic_network)
            agents.append(agent)

        return agents

    def teach_agents(self, training_cfg, experiment_folder=''):
        training = experiment_folder == ''

        n_episodes = training_cfg['n_episodes']
        timeout = training_cfg['timeout']

        self.experiment_logger.create_experiment(self.agent_name,
                                                 self.env_cfg['name'],
                                                 training_cfg,
                                                 experiment_folder)  # this is a wraapper over summarywriter()

        for episode in range(n_episodes):
            state = self.env.reset()
            step = 0
            while True:
                actions = self.step_agents(state)
                state, reward, episode_end, info = self.env.step(actions)
                losses = self.update_agents(reward, episode_end, state)

                self.experiment_logger.add_agent_scalars('losses', losses, track_mean=True)
                self.experiment_logger.add_agent_scalars('reward', reward, track_mean=True, track_sum=True, log=True)

                if (self.is_episodic and episode_end) or (not self.is_episodic and (step + 1) % self.log_freq == 0):
                    self.experiment_logger.log_progress(episode, step)
                    self.experiment_logger.add_agent_scalars('episode_length', data=step, step=episode, log=True)

                if step > timeout or (self.is_episodic and episode_end):
                    break

                step += 1

    def step_agents(self, state):
        if self.n_agents == 1:
            return self.agents[0].step(state)
        else:
            return [self.agents[key].step(state[key]) for key in self.agent_keys]

    def update_agents(self, reward, episode_end, new_state):
        if self.n_agents == 1:
            loss = self.agents[0].update_policy(reward, episode_end, new_state)
        else:
            loss = []
            for key in self.agent_keys:
                loss.append(self.agents[key].update_policy(
                    reward[key], episode_end[key], new_state[key]
                ))
        return loss


@register_controller
class ACController(BaseController):
    def __init__(self, env_cfg, cfg):
        self.ac_name = cfg.get('ac_network', None)
        self.actor_name = cfg.get('actor_network', None)
        self.critic_name = cfg.get('critic_network', None)
        self.ac_cfg = cfg.get('ac', cfg['actor'])
        self.actor_cfg = self.ac_cfg
        self.critic_cfg = cfg.get('critic', None)
        super(ACController, self).__init__(env_cfg, cfg)


@register_controller
class PGController(BaseController):
    def __init__(self, env_cfg, cfg):
        self.actor_name = cfg.get('actor_network', None)
        self.actor_cfg = cfg['actor']
        super(PGController, self).__init__(env_cfg, cfg)

    def make_agents(self):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            n_features = self.env.observation_space.shape[0]
        else:
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                n_features = self.env.observation_space.n
            else:
                raise NotImplementedError
        n_actions = self.env.action_space.n
        agents = []
        for i in range(0, self.n_agents):
            actor_network = get_network(key=self.actor_name,
                                        out_shape=n_actions,
                                        cfg=self.actor_cfg,
                                        n_features=n_features)
            agent = AGENT_REGISTRY[self.agent_name](self.is_episodic,
                                                    self.cfg,
                                                    actor_network)
            agents.append(agent)

        return agents
