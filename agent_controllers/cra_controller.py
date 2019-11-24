from agent_controllers.factory import register_controller
from networks.base_networks import *
from agent_algorithms.factory import AGENT_REGISTRY
from networks.factory import get_network
from agent_controllers.base_controllers import BaseController


@register_controller
class CRAController(BaseController):
    def __init__(self, env_cfg, cfg):
        self.actor_name = cfg.get('actor_network')
        self.actor_cfg = cfg.get('actor')
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
        agents = []
        for i in range(0, self.n_agents):
            ac_network = get_network(key=self.actor_name,
                                     out_shape=n_actions,
                                     out_shape2=critic_estimates,
                                     cfg=self.actor_cfg,
                                     n_features=n_features)

            agent = AGENT_REGISTRY[self.agent_name](self.is_episodic,
                                                    ac_network, self.cfg)
            agents.append(agent)

        return agents
