import gym
import torch
import torch.nn.functional as F

import envs.env_factory as env_factory
import utils.experiment_utils as experiment_utils
import utils.model_utils as model_utils
from envs.training.env_encoder_trainers import EnvEncoderTrainer
from networks.net_factory import get_network
from networks.network_interface import NetworkInterface


class EnvTraining(object):
    class LoggingMetrics(object):
        mean_abs_diff = 'img_encoder/mean_abs_diff'
        loss = 'img_encoder/loss'

    def __init__(self, env_cfg, network_cfg):
        training: EnvEncoderTrainer = env_factory.get_state_trainer(env_cfg)
        obs_space = training.get_in_spaces()
        out_space = training.calc_out_space()
        in_shapes = model_utils.space_to_shapes(obs_space)
        out_shapes = model_utils.space_to_shapes(out_space)
        self.network: NetworkInterface = get_network(network_cfg, in_shapes, out_shapes=out_shapes)

    def verify(self):
        pass

    def train(self, training_cfg):
        logger: experiment_utils.ExperimentLogger = experiment_utils.ExperimentLogger()
        logger.create_experiment(algo=self.network, env_name=self.env_cfg['name'],
                                 training_cfg=training_cfg)

        checkpoint_freq = training_cfg['checkpoint_freq']
        batch_size = training_cfg['batch_size']
        out_space = self.calc_out_space()
        assert isinstance(out_space, gym.spaces.Box), 'assuming we are predicting coordinates'

        net_trainer = self.network.create_optimizer()

        predictor = \
            torch.nn.Linear(in_features=self.network.get_out_features(), out_features=self.y_shape)
        self.network.pretrain(predictor)
        full_network = torch.nn.Sequential(self.network, predictor)

        for i in range(0, 1000):
            new_batch, y_true = self.generate_batch(batch_size=batch_size)
            out = self.network.forward(new_batch)
            out = predictor(out)
            critic_loss = F.smooth_l1_loss(input=out, target=y_true,
                                           reduction='mean')  # .5 * advantage.pow(2).mean()
            logger.add_agent_scalars(label=self.LoggingMetrics.loss, data=critic_loss.cpu().item(), log=True)
            # critic_loss = F.mse_loss(input=out, target=y_true, reduction='mean')  # .5 * advantage.pow(2).mean()
            # mse_loss = (out - y_true).pow(2).mean()
            critic_loss.backward(retain_graph=True)
            net_trainer.update_parameters()
            if ((i + 1) % checkpoint_freq) == 0:
                mean_abs_diff = self.mean_abs_diff(full_network)
                print('------')
                print(y_true)
                print(out)
                print('------')
                logger.add_agent_scalars(label=self.LoggingMetrics.mean_abs_diff, data=mean_abs_diff, log=True)
            print(i)
            # print(critic_loss)
            # print(mse_loss)
            # print('----')
