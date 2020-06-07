import utils.model_utils as model_utils
from agent_algorithms.factory import AGENT_REGISTRY
from agent_controllers.base_controllers import BaseController
from agent_controllers.factory import register_controller
from networks.factory import get_network


@register_controller
class ACController(BaseController):
    def __init__(self, env_cfg, cfg):
        self.ac_name = cfg.get('ac_network', None)
        self.actor_name = cfg.get('actor_network', None)
        self.critic_name = cfg.get('critic_network', None)
        self.ac_cfg = cfg.get('ac', cfg['actor'])
        self.share_parameters = self.ac_name is not None
        self.actor_cfg = self.ac_cfg
        self.critic_cfg = cfg.get('critic', None)
        super(ACController, self).__init__(env_cfg, cfg)

    def make_agents(self):
        in_shapes = model_utils.spaces_to_shapes(self.env.observation_space)
        action_shapes = model_utils.spaces_to_shapes(
            self.env.action_space)  # n_actions, can add to output shapes in controller
        critic_estimates = [(1,), ]  # value estimator
        agents = []
        for i in range(0, self.n_agents):
            if self.share_parameters:

                ac_network = get_network(key=self.ac_name,
                                         in_shapes=in_shapes,
                                         out_shapes=action_shapes,
                                         cfg=self.ac_cfg)
                agent = AGENT_REGISTRY[self.agent_name](self.is_episodic,
                                                        self.cfg,
                                                        ac_network)
            else:
                actor_network = get_network(key=self.actor_name,
                                            in_shapes=in_shapes,
                                            out_shapes=action_shapes,
                                            cfg=self.actor_cfg)
                critic_network = get_network(key=self.critic_name,
                                             in_shapes=in_shapes,
                                             out_shapes=critic_estimates,
                                             cfg=self.critic_cfg)
                agent = AGENT_REGISTRY[self.agent_name](self.is_episodic,
                                                        self.cfg,
                                                        actor_network,
                                                        critic_network)
            agents.append(agent)

        return agents
