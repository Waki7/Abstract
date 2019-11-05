from agent_controllers.factory import register_controller
from networks.base_networks import *
from agent_algorithms.factory import AGENT_REGISTRY
from networks.factory import NETWORK_REGISTERY
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
        self.ac_name = cfg.get('ac_network', None)
        self.actor_name = cfg.get('actor_network', None)
        self.critic_name = cfg.get('critic_network', None)
        self.ac_cfg = self.cfg.get('ac', None)
        self.actor_cfg = self.cfg.get('actor', None)
        self.critic_cfg = self.cfg.get('critic', None)
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
                ac_network = NETWORK_REGISTERY[self.ac_name](n_features,
                                                             n_actions,
                                                             critic_estimates,
                                                             self.ac_cfg)
                actor_network = ac_network.actor
                critic_network = ac_network.critic
            else:
                actor_network = NETWORK_REGISTERY[self.actor_name](n_features,
                                                                   n_actions,
                                                                   self.actor_cfg)
                critic_network = NETWORK_REGISTERY[self.critic_name](n_features,
                                                                     critic_estimates,
                                                                     self.critic_cfg)
            agent = AGENT_REGISTRY[self.agent_name](self.env,
                                                    actor_network,
                                                    critic_network,
                                                    self.is_episodic,
                                                    self.cfg)
            agents.append(agent)

        return self.cfg['agents']

    def teach_agents(self, training_cfg, experiment_folder=''):
        training = experiment_folder == ''
        max_episodes = cfg.experiment.MAX_EPISODES if self.is_episodic else 1

        experiment_writer = self.experiment_logger.create_experiment(self.cfg['name'],
                                                                     self.env_cfg['name'],
                                                                     training_cfg,
                                                                     training)  # this is a wraapper over summarywriter()

        for episode in range(max_episodes):
            state = self.env.reset()
            step = 0
            while True:
                actions = self.step_agents(state)
                state, reward, episode_end, info = self.env.step(actions)
                losses = self.update_agents(reward, episode_end, state)

                experiment_writer.add_scalar('losses', losses)
                experiment_writer.add_scalar('reward', reward)

                if (self.is_episodic and episode_end) or (not self.is_episodic and (step + 1) % self.log_freq == 0):
                    experiment_writer.log_progress(episode, step)

                if step > training_cfg['timeout'] or (self.is_episodic and episode_end):
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
class IACController(BaseController):
    def __init__(self, env_cfg, cfg):
        super().__init__(env_cfg, cfg)
