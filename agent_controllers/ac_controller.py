import networks.net_factory as net_factory
import utils.experiment_utils as exp_utils
import utils.model_utils as model_utils
from agent_algorithms.factory import AGENT_REGISTRY
from agent_controllers.base_controllers import BaseController
from agent_controllers.factory import register_controller


@register_controller
class ACController(BaseController):
    def __init__(self, env_cfg, cfg):
        self.cfg = {}
        self.ac_name = exp_utils.copy_config_param(src_cfg=cfg, target_cfg=self.cfg, param_name='ac_network')
        self.actor_name = exp_utils.copy_config_param(src_cfg=cfg, target_cfg=self.cfg, param_name='actor_network')
        self.critic_name = exp_utils.copy_config_param(src_cfg=cfg, target_cfg=self.cfg, param_name='critic_network')
        self.image_encoder_name = exp_utils.copy_config_param(src_cfg=cfg, target_cfg=self.cfg,
                                                              param_name='image_encoder_network', fallback_value=None)

        self.ac_cfg = exp_utils.copy_config_param(src_cfg=cfg, target_cfg=self.cfg, param_name='ac',
                                                  fallback_value=cfg['actor'])
        self.share_parameters = self.ac_name is not None
        self.actor_cfg = self.ac_cfg
        self.critic_cfg = exp_utils.copy_config_param(src_cfg=cfg, target_cfg=self.cfg, param_name='critic',
                                                      fallback_value=self.ac_cfg)
        self.image_encoder_cfg = exp_utils.copy_config_param(src_cfg=cfg, target_cfg=self.cfg,
                                                             param_name='image_encoder',
                                                             fallback_value={})
        self.image_encoder = None

        super(ACController, self).__init__(env_cfg, cfg)

    def generate_agent(self, in_shapes, action_shapes, critic_estimates):
        if self.share_parameters:
            ac_network = net_factory.get_network(key=self.ac_name,
                                                 in_shapes=in_shapes,
                                                 out_shapes=action_shapes,
                                                 cfg=self.ac_cfg)
            agent = AGENT_REGISTRY[self.agent_name](self.is_episodic,
                                                    self.cfg,
                                                    ac_network)
        else:
            actor_network = net_factory.get_network(key=self.actor_name,
                                                    in_shapes=in_shapes,
                                                    out_shapes=action_shapes,
                                                    cfg=self.actor_cfg)
            critic_network = net_factory.get_network(key=self.critic_name,
                                                     in_shapes=in_shapes,
                                                     out_shapes=critic_estimates,
                                                     cfg=self.critic_cfg)
            agent = AGENT_REGISTRY[self.agent_name](self.is_episodic,
                                                    self.cfg,
                                                    actor=actor_network,
                                                    critic=critic_network)
        return agent

    def make_agents(self):
        in_shapes = model_utils.spaces_to_shapes(self.env.observation_space)
        print(in_shapes)
        action_shapes = model_utils.spaces_to_shapes(
            self.env.action_space)  # n_actions, can add to output shapes in controller
        critic_estimates = [(1,), ]  # value estimator
        if self.image_encoder_name is not None:
            # TODO fix the shape check for if we add language
            img_shapes = [shape for shape in in_shapes if len(shape) > 1]
            print(img_shapes)
            assert len(img_shapes) == 1, 'not supporting multiple images at the moment'
            img_encoder_out_shapes = [(self.image_encoder_cfg['out_features'],)]
            self.image_encoder = net_factory.get_network(key=self.image_encoder_name,
                                                         cfg=self.image_encoder_cfg, in_shapes=img_shapes,
                                                         out_shapes=img_encoder_out_shapes)

            in_shapes = [shape for shape in in_shapes if len(shape) < 2]
            in_shapes.extend(img_encoder_out_shapes)
        print(in_shapes)
        print(exit(9))
        agents = []
        for i in range(0, self.n_agents):
            new_agent = self.generate_agent(in_shapes=in_shapes, action_shapes=action_shapes,
                                            critic_estimates=critic_estimates)
            agents.append(new_agent)

        return agents

    def step_agent(self, agent, batched_obs):
        batched_obs = self.state_encoder.forward(batched_obs)
        actions = agent.step(batched_obs)
        return actions
