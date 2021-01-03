from typing import *

import gym
import numpy as np
import torch
import torch.nn.functional as F

import envs.env_factory as env_factory
import utils.experiment_utils as experiment_utils
import utils.model_utils as model_utils
from envs.training.base_env_encoding_task import EnvEncodingTask
from networks.net_factory import get_network
from networks.network_interface import NetworkInterface


class EnvEncodingController(object):
    class LoggingMetrics(object):
        mean_abs_diff = 'img_encoder/mean_abs_diff'
        loss = 'img_encoder/loss'

    def __init__(self, env_cfg, network_cfg):
        self.network_cfg = network_cfg
        self.env_cfg = env_cfg
        trainer: EnvEncodingTask = env_factory.get_state_trainer(env_cfg)
        obs_space: gym.spaces.Space = trainer.get_in_spaces()
        out_space: gym.spaces.Space = trainer.calc_out_space()
        self.trainer = trainer
        assert isinstance(out_space, gym.spaces.Box)
        self.y_shape = np.prod(out_space.shape)
        self.network: NetworkInterface = get_network(network_cfg,
                                                     obs_space=obs_space,
                                                     out_space=out_space)

    def verify(self):
        pass

    def config_indices(self, training_cfg) -> Dict[str, Dict]:
        return {
            'training_cfg': training_cfg,
            'env_cfg': self.env_cfg,
            'network_cfg': self.network_cfg
        }

    def eval_metrics(self):
        new_batch, y_true = self.trainer.generate_batch()
        with torch.no_grad():
            y_pred = self.network.forward(new_batch)
        return model_utils.mean_abs_diff(y_pred=y_pred, y_true=y_true)

    def train(self, training_cfg):
        logger: experiment_utils.ExperimentLogger = experiment_utils.ExperimentLogger()
        logger.create_experiment(algo=self.network,
                                 run_dir='training',
                                 config_dict=self.config_indices(training_cfg))

        checkpoint_freq = training_cfg['checkpoint_freq']
        batch_size = training_cfg['batch_size']
        out_space: gym.spaces.Space = self.trainer.calc_out_space()
        assert isinstance(out_space,
                          gym.spaces.Box), 'assuming we are predicting coordinates'

        net_trainer = self.network.create_optimizer()
        predictor = \
            torch.nn.Linear(in_features=self.network.get_out_features(),
                            out_features=self.y_shape)
        self.network.add_temp_predictor(predictor)
        full_network = torch.nn.Sequential(self.network, predictor)

        for i in range(0, 1000):
            new_batch, y_true = self.trainer.generate_batch(
                batch_size=batch_size)
            out = full_network(new_batch)
            critic_loss = F.smooth_l1_loss(input=out, target=y_true,
                                           reduction='mean')  # .5 * advantage.pow(2).mean()
            logger.add_agent_scalars(label=self.LoggingMetrics.loss,
                                     data=critic_loss.cpu().item(), log=True)
            # critic_loss = F.mse_loss(input=out, target=y_true, reduction='mean')  # .5 * advantage.pow(2).mean()
            # mse_loss = (out - y_true).pow(2).mean()
            critic_loss.backward(retain_graph=True)
            net_trainer.update_parameters()
            if ((i + 1) % checkpoint_freq) == 0:
                mean_abs_diff = self.eval_metrics()
                print('------')
                print(y_true)
                print(out)
                print('------')
                logger.add_agent_scalars(
                    label=self.LoggingMetrics.mean_abs_diff,
                    data=mean_abs_diff, log=True)
            print(i)
            # print(critic_loss)
            # print(mse_loss)
            # print('----')
