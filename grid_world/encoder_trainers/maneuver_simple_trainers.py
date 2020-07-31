import gym
import numpy as np

import grid_world.encoder_trainers as enc_trainers
import grid_world.envs as envs

import utils.model_utils as model_utils
import torch
import torch.nn.functional as F

class StateEncodingProtocol(enc_trainers.EnvEncoderTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.env: envs.ManeuverSimple

    def get_in_spaces(self) -> gym.spaces:
        return self.env.get_obs_space()

    def get_out_spaces(self) -> gym.spaces:
        n_landmarks = self.env.n_landmarks
        n_agents = self.env.n_agents
        n_foreign_friendlies = self.env.n_foreign_enemies
        n_foreign_enemies = self.env.n_foreign_enemies
        bounds = self.env.world.bounds
        n_dim = len(bounds)

        n_total_objects = n_landmarks + n_agents + n_foreign_enemies + n_foreign_friendlies
        lower_bounds = [interval[0] for interval in bounds]
        upper_bounds = [interval[1] for interval in bounds]

        return gym.spaces.Box(low=np.min(lower_bounds), high=np.max(upper_bounds), shape=(n_total_objects, n_dim))

    def get_classifier(self):
        pass

    def generate_batch(self):
        inputs = []
        for i in range(0, 100):
            state = self.env.reset()
            inputs.append(state)
        batched_obs = model_utils.batch_env_observations(inputs, self.env.observation_space)
        batched_obs = model_utils.scale_space(state=batched_obs, space=self.env.observation_space)
        batched_obs = model_utils.list_to_torch_device(batched_obs)
        return batched_obs

    def train(self, network):
        network_features = network.get_out_features()
        out_space = self.get_out_spaces()

        classifier = model_utils.get_activation_for_space(out_space, in_features=network_features)
        network.pretrain(classifier)

        for i in range(0, 10):
            new_batch = self.generate_batch()
            out = network.forward(new_batch[0])
            critic_loss = (F.smooth_l1_loss(input=out, target=discounted_rewards_vec,
                                            reduction='none') * zero_done_mask).mean()  # .5 * advantage.pow(2).mean()
            print(out.shape)
            print(exit(9))
