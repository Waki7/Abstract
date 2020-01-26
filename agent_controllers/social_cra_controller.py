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
