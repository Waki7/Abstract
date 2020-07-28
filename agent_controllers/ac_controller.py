from typing import Union, Dict, List

import torch

import networks.net_factory as net_factory
import networks.network_interface as nets
import utils.experiment_utils as exp_utils
import utils.model_utils as model_utils
from agent_algorithms.factory import AGENT_REGISTRY
from agent_controllers.base_controllers import BaseController
from agent_controllers.factory import register_controller


@register_controller
class ACController(BaseController):
    def __init__(self, env_cfg, cfg):
        self.cfg = {}
        self.ac_cfg = exp_utils.copy_config_param(src_cfg=cfg, target_cfg=self.cfg, param_name='ac',
                                                  fallback_value=cfg['actor'])
        self.share_parameters = 'critic' not in cfg
        print(self.share_parameters)
        print(exit(9))
        self.actor_cfg = self.ac_cfg
        self.critic_cfg = exp_utils.copy_config_param(src_cfg=cfg, target_cfg=self.cfg, param_name='critic',
                                                      fallback_value=self.ac_cfg)
        self.image_encoder_cfg = exp_utils.copy_config_param(src_cfg=cfg, target_cfg=self.cfg,
                                                             param_name='image_encoder',
                                                             fallback_value={})

        self.ac_name = self.ac_cfg.get('name', '')
        self.actor_name = self.actor_cfg.get('name', '')
        self.critic_name = self.critic_cfg.get('name', '')

        self.image_encoder_name = self.image_encoder_cfg.get('name', '')

        self.train_image_encoder = self.image_encoder_cfg.get('train')


        self.image_encoder = None

        self.image_feature_idxs = []
        self.planning_feature_idxs = []
        super(ACController, self).__init__(env_cfg, cfg)

    def assign_feature_idxs(self, shapes):
        self.image_feature_idxs = []
        self.planning_feature_idxs = []

        for idx, shape in enumerate(shapes):
            if len(shape) == 3:
                self.image_feature_idxs.append(idx)
            else:
                self.planning_feature_idxs.append(idx)

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
        planner_in_shapes = model_utils.spaces_to_shapes(self.env.observation_space)
        self.assign_feature_idxs(planner_in_shapes)
        action_shapes = model_utils.spaces_to_shapes(
            self.env.action_space)  # n_actions, can add to output shapes in controller
        critic_estimates = [(1,), ]  # value estimator
        if self.image_encoder_name is not None:
            # TODO fix the shape check for if we add language
            image_encoder_in_shapes = model_utils.get_idxs_of_list(list=planner_in_shapes, idxs=self.image_feature_idxs)
            assert len(image_encoder_in_shapes) == 1, 'not supporting multiple images at the moment'
            self.image_encoder: nets.NetworkInterface = net_factory.get_network(key=self.image_encoder_name,
                                                                                cfg=self.image_encoder_cfg,
                                                                                in_shapes=image_encoder_in_shapes)
            img_encoder_out_shapes = self.image_encoder.get_out_shapes()
            self.image_encoder.train() if self.train_image_encoder else self.image_encoder.eval()

            planner_in_shapes = model_utils.get_idxs_of_list(list=planner_in_shapes, idxs=self.planning_feature_idxs)
            planner_in_shapes.extend(img_encoder_out_shapes)
        agents = []
        for i in range(0, self.n_agents):
            new_agent = self.generate_agent(in_shapes=planner_in_shapes, action_shapes=action_shapes,
                                            critic_estimates=critic_estimates)
            agents.append(new_agent)

        return agents

    def step_agent(self, agent, batched_obs):
        if self.image_encoder is not None:
            image_obs = model_utils.get_idxs_of_list(list=batched_obs, idxs=self.image_feature_idxs)[0]
            if self.train_image_encoder:
                image_embedding = self.image_encoder.forward(image_obs)

            else:
                with torch.no_grad():
                    image_embedding = self.image_encoder.forward(image_obs).detach()
            batched_obs = model_utils.get_idxs_of_list(list=batched_obs, idxs=self.planning_feature_idxs)
            batched_obs.append(image_embedding)
        actions = agent.step(batched_obs)
        return actions

    def update_agents(self, rewards: Union[List[float], Dict],
                      episode_ends: Union[List[bool], Dict],
                      is_batch_env):
        batch_reward, batch_end = self.convert_env_feedback_for_agent(rewards=rewards,
                                                                      episode_ends=episode_ends,
                                                                      is_batch_env=is_batch_env)
        if self.n_agents == 1:
            loss = self.agents[0].update_policy(batch_reward, batch_end)
            if loss is not None and len(loss) > 0 and self.image_encoder is not None and self.train_image_encoder:
                self.image_encoder.update_parameters()
        else:
            raise NotImplementedError('NEED TO UPDATE ENVIRONMENT OBSERVATION SPACES FOR THE NEW_STATE')
            # agent_reward = []
            # agent_ends = []
            # agent_state = []
            #
            # loss = {}
            # for key in self.agent_keys:
            #     loss[key] = (self.agents[key].update_policy(
            #         reward[key], episode_end[key], new_state[key]
            #     ))
        return loss
    
    def load(self):
        pass
    
    def save(self):
        pass
