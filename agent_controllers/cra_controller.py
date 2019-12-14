import gym

from agent_algorithms.factory import AGENT_REGISTRY
from agent_controllers.base_controllers import BaseController
from agent_controllers.factory import register_controller
from networks.factory import get_network


@register_controller
class CRAController(BaseController):
    def __init__(self, env_cfg, cfg):
        self.actor_name = cfg.get('actor_network')
        self.actor_cfg = cfg.get('actor_cfg')
        self.critic_name = cfg.get('critic_network')
        self.critic_cfg = cfg.get('critic_cfg', self.actor_cfg)
        self.share_parameters = self.critic_name == None
        super(CRAController, self).__init__(env_cfg, cfg)

    def make_agents(self):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            n_features = self.env.observation_space.shape[0]
        else:
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                n_features = self.env.observation_space.n
            else:
                raise NotImplementedError
        n_actions = self.env.action_space.n
        critic_estimates = 1
        aux_estimates = 1
        agents = []
        for i in range(0, self.n_agents):
            if self.share_parameters:
                ac_network = get_network(key=self.actor_name,
                                         out_shape=n_actions,
                                         out_shape2=critic_estimates,
                                         out_shape3=aux_estimates,
                                         cfg=self.actor_cfg,
                                         n_features=n_features)

                agent = AGENT_REGISTRY[self.agent_name](is_episodic=self.is_episodic,
                                                        actor=ac_network,
                                                        cfg=self.cfg)
            else:

                actor_network = get_network(key=self.actor_name,
                                            out_shape=n_actions,
                                            cfg=self.actor_cfg,
                                            n_features=n_features)

                critic_network = get_network(key=self.critic_name,
                                             out_shape=critic_estimates,
                                             out_shape2=aux_estimates,
                                             cfg=self.critic_cfg,
                                             n_features=n_features)

                agent = AGENT_REGISTRY[self.agent_name](is_episodic=
                                                        self.is_episodic,
                                                        actor=actor_network,
                                                        critic=critic_network,
                                                        cfg=self.cfg)
            agents.append(agent)

        return agents


@register_controller
class EXPController(BaseController):
    def __init__(self, env_cfg, cfg):
        self.actor_name = cfg.get('actor_network')
        self.actor_cfg = cfg.get('actor_cfg')
        self.critic_name = cfg.get('critic_network')
        self.critic_cfg = cfg.get('critic_cfg')
        self.reward_network_name = cfg.get('reward_network')
        self.reward_network_cfg = cfg.get('reward_cfg', {})
        super(EXPController, self).__init__(env_cfg, cfg)

    def make_agents(self):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            n_features = self.env.observation_space.shape[0]
        else:
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                n_features = self.env.observation_space.n
            else:
                raise NotImplementedError
        n_actions = self.env.action_space.n
        critic_estimates = 1
        aux_estimates = 1
        agents = []
        for i in range(0, self.n_agents):
            actor_network = get_network(key=self.actor_name,
                                        out_shape=n_actions,
                                        cfg=self.actor_cfg,
                                        n_features=n_features)

            critic_network = get_network(key=self.critic_name,
                                         out_shape=critic_estimates,
                                         out_shape2=aux_estimates,
                                         cfg=self.critic_cfg,
                                         n_features=n_features)
            reward_network = get_network(key=self.reward_network_name,
                                         n_features=2,
                                         out_shape=1,
                                         cfg=self.reward_network_cfg)

            agent = AGENT_REGISTRY[self.agent_name](is_episodic=
                                                    self.is_episodic,
                                                    actor=actor_network,
                                                    critic=critic_network,
                                                    cfg=self.cfg,
                                                    attention=reward_network)
            agents.append(agent)

        return agents
