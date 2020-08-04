import gym
import numpy as np
import torch
import torch.nn.functional as F

import grid_world.encoder_trainers as enc_trainers
import grid_world.envs as envs
import networks.network_interface as nets
import settings
import utils.model_utils as model_utils


class StateEncodingProtocol(enc_trainers.EnvEncoderTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.env: envs.ManeuverSimple
        self.in_space = self.get_in_spaces()
        self.out_space = self.calc_out_space()
        self.y_shape = np.prod(self.out_space.shape)

    def get_in_spaces(self) -> gym.spaces:
        return self.env.calc_observation_space()

    def get_total_objects(self):
        n_total_objects = self.env.n_landmarks + self.env.n_agents + self.env.n_foreign_enemies + self.env.n_foreign_enemies
        return n_total_objects

    def get_n_dims(self):
        return len(self.env.world.bounds)

    def calc_out_space(self) -> gym.spaces.Box:
        bounds = self.env.world.bounds

        n_dim = self.get_n_dims()
        n_total_objects = self.get_total_objects()

        lower_bounds = [interval[0] for interval in bounds]
        upper_bounds = [interval[1] for interval in bounds]

        return gym.spaces.Box(low=np.min(lower_bounds), high=np.max(upper_bounds), shape=(n_total_objects, n_dim))

    def generate_batch(self, batch_size=100) -> torch.Tensor:
        inputs = []
        y_trues = []
        for i in range(0, batch_size):
            state: np.ndarray = self.env.reset()
            inputs.append(state)
            coordinates = np.asarray(self.env.get_object_coordinates()).flatten()
            y_trues.append(coordinates)

        batched_obs = model_utils.batch_env_observations(inputs, self.in_space)
        batched_obs = model_utils.scale_space(state=batched_obs, space=self.in_space)
        batched_obs = model_utils.list_to_torch_device(batched_obs)[0]

        y_trues = np.stack(y_trues)
        y_trues = model_utils.scale_space(state=y_trues, space=self.out_space)
        y_trues = torch.tensor(y_trues).to(**settings.ARGS)

        return batched_obs, y_trues

    def validate(self, network):
        new_batch, y_true = self.generate_batch(1)
        with torch.no_grad():
            out = network.forward(new_batch)
        print('expected {}'.format(y_true))
        print('actual prediction {}'.format(out))
        print('average_diff {}'.format(torch.abs(out-y_true).mean()))

    def train(self, encoder: nets.NetworkInterface, training_cfg):
        checkpoint_freq = training_cfg['checkpoint_freq']
        out_space = self.calc_out_space()
        assert isinstance(out_space, gym.spaces.Box), 'assuming we are predicting coordinates'

        net_trainer = encoder.create_optimizer()

        predictor = \
            torch.nn.Linear(in_features=encoder.get_out_features(), out_features=self.y_shape)
        encoder.pretrain(predictor)
        full_network = torch.nn.Sequential(encoder, predictor)

        for i in range(0, 1000):
            new_batch, y_true = self.generate_batch()
            out = full_network.forward(new_batch)
            critic_loss = F.smooth_l1_loss(input=out, target=y_true, reduction='mean')  # .5 * advantage.pow(2).mean()
            # critic_loss = F.mse_loss(input=out, target=y_true, reduction='mean')  # .5 * advantage.pow(2).mean()
            # mse_loss = (out - y_true).pow(2).mean()
            critic_loss.backward()
            net_trainer.update_parameters()
            if ((i + 1) % checkpoint_freq) == 0:
                self.validate(full_network)
            print(i)
            print(critic_loss)
            # print(mse_loss)
            print('----')
