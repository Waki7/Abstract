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

    def generate_batch(self):
        inputs = []
        y_trues = []
        for i in range(0, 100):
            state: np.ndarray = self.env.reset()
            inputs.append(state)
            coordinates = np.asarray(self.env.get_object_coordinates()).flatten()
            y_trues.append(coordinates)

        batched_obs = model_utils.batch_env_observations(inputs, self.in_space)
        batched_obs = model_utils.scale_space(state=batched_obs, space=self.env.observation_space)
        batched_obs = model_utils.list_to_torch_device(batched_obs)[0]

        y_trues = np.stack(y_trues)
        y_trues = torch.tensor(y_trues).to(**settings.ARGS)

        return batched_obs, y_trues

    def train(self, network: nets.NetworkInterface):
        out_space = self.calc_out_space()
        assert isinstance(out_space, gym.spaces.Box), 'assuming we are predicting coordinates'

        net_trainer = network.create_optimizer()

        classifier = torch.nn.Linear(in_features=network.get_out_features(), out_features=self.y_shape)
        network.pretrain(classifier)

        for i in range(0, 10):
            new_batch, y_true = self.generate_batch()
            out = classifier.forward(network.forward(new_batch))

            critic_loss = F.smooth_l1_loss(input=out, target=y_true, reduction='mean')  # .5 * advantage.pow(2).mean()
            critic_loss.backward()
            net_trainer.update_parameters()
            print(i)
            print(critic_loss)
            print('----')
