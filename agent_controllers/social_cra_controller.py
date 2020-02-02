import gym

from agent_algorithms.factory import AGENT_REGISTRY
from agent_controllers.base_controllers import BaseController
from agent_controllers.factory import register_controller
from networks.factory import get_network


@register_controller
class SocialCRAController(BaseController):
    def __init__(self, env_cfg, cfg):
        self.ac_name = cfg.get('ac_network')
        self.ac_cfg = cfg.get('ac_cfg')
        self.concepts = cfg.get('n_concepts')
        super(SocialCRAController, self).__init__(env_cfg, cfg)

    def make_agents(self):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            n_features = self.env.observation_space.shape[0]
        else:
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                n_features = self.env.observation_space.n
            else:
                raise NotImplementedError
        n_actions = self.env.action_space.n
        # n_features += n_actions

        critic_estimates = 1
        aux_estimates = 1
        agents = []

        # teacher agents
        for i in range(0, self.n_agents):
            ac_network = get_network(key=self.ac_name,
                                     out_shape=n_actions,
                                     out_shape2=critic_estimates,
                                     out_shape3=aux_estimates,
                                     cfg=self.ac_cfg,
                                     n_features=n_features)

            agent = AGENT_REGISTRY[self.agent_name](is_episodic=self.is_episodic,
                                                    ac=ac_network,
                                                    cfg=self.cfg)

            agents.append(agent)

        # teacher agents
        for i in range(0, self.n_agents):
            ac_network = get_network(key=self.ac_name,
                                     out_shape=n_actions,
                                     out_shape2=critic_estimates,
                                     out_shape3=aux_estimates,
                                     cfg=self.ac_cfg,
                                     n_features=n_features)

            agent = AGENT_REGISTRY[self.agent_name](is_episodic=self.is_episodic,
                                                    ac=ac_network,
                                                    cfg=self.cfg)

            agents.append(agent)

        return agents

    def think_agents(self):
        think_steps = 20
        if self.n_agents == 1:
            return self.agents[0].think(state, think_steps)
        else:
            return [self.agents[key].think(state[key], think_steps) for key in self.agent_keys]

    def teach_agents(self, training_cfg, experiment_folder=''):
        training = experiment_folder == ''

        n_episodes = training_cfg['n_episodes']
        think_freq = training_cfg.get('think_freq', 20)

        timeout = training_cfg['timeout']

        self.experiment_logger.create_experiment(self.agent_name,
                                                 self.env_cfg['name'],
                                                 training_cfg,
                                                 experiment_folder,
                                                 env_cfg=self.env_cfg,
                                                 agent_cfg=self.cfg,
                                                 )  # this is a wraapper over summarywriter()

        for episode in range(n_episodes):
            state = self.env.reset()
            step = 0
            while True:
                actions = self.step_agents(state)
                state, reward, episode_end, info = self.env.step(actions)
                losses = self.update_agents(reward, episode_end, state)

                self.experiment_logger.add_agent_scalars('losses', losses, track_mean=True)
                self.experiment_logger.add_agent_scalars('reward', reward, track_mean=True, track_sum=True, log=True)

                if (step + 1) % think_freq == 0:
                    self.think_agents()

                if (self.is_episodic and episode_end) or (not self.is_episodic and (step + 1) % self.log_freq == 0):
                    self.experiment_logger.log_progress(episode, step)
                    self.experiment_logger.add_agent_scalars('episode_length', data=step, step=episode, log=True)

                if step > timeout or (self.is_episodic and episode_end):
                    break

                step += 1
