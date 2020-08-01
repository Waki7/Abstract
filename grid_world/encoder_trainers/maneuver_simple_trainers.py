import typing as typ

import gym
import numpy as np
import torch.nn.functional as F

import grid_world.encoder_trainers as enc_trainers
import grid_world.envs as envs
import utils.model_utils as model_utils


class StateEncodingProtocol(enc_trainers.EnvEncoderTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.env: envs.ManeuverSimple
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
            state: typ.List[np.ndarray] = self.env.reset()
            inputs.append(state)
            coordinates = np.asarray(self.env.get_object_coordinates())

        batched_obs = model_utils.list_to_torch_device(inputs)
        batched_obs = model_utils.batch_env_observations(batched_obs, self.env.observation_space)
        batched_obs = model_utils.scale_space(state=batched_obs, space=self.env.observation_space)



        return batched_obs, y_true

    def train(self, network):
        network_features = network.get_out_features()
        out_space = self.calc_out_space()

        classifier = model_utils.get_activation_for_space(out_space, in_features=network_features)
        network.pretrain(classifier)

        for i in range(0, 10):
            new_batch = self.generate_batch()
            out = network.forward(new_batch[0])
            critic_loss = (F.smooth_l1_loss(input=out, target=discounted_rewards_vec,
                                            reduction='none') * zero_done_mask).mean()  # .5 * advantage.pow(2).mean()
            print(out.shape)
            print(exit(9))
