from typing import *
import gym
import numpy as np
import torch
import torch.nn.functional as F

import networks.network_interface as nets
import settings
import utils.experiment_utils as experiment_utils
from envs.env_factory import get_env
import utils.model_utils as model_utils
import envs.grid_world.env_variations.grid_world as grid_world
from envs.training.base_env_encoding_task import EnvEncodingTask


class GridEncodingTask(EnvEncodingTask):
    def __init__(self, env_cfg):
        super().__init__(env_cfg)
        self.env: grid_world.GridEnv = get_env(env_cfg=env_cfg)
        self.in_space = self.get_in_spaces()
        self.out_space = self.calc_out_space()

    def get_in_spaces(self) -> gym.spaces:
        return self.env.calc_observation_space()

    def get_total_objects(self):
        n_total_objects = self.env.n_landmarks + self.env.n_agents + \
                          self.env.n_foreign_enemies + \
                          self.env.n_foreign_enemies
        return n_total_objects

    def get_n_dims(self):
        return len(self.env.world.bounds)

    def calc_out_space(self) -> gym.spaces.Box:
        bounds = self.env.world.bounds

        n_dim = self.get_n_dims()
        n_total_objects = self.get_total_objects()

        lower_bounds = [interval[0] for interval in bounds]
        upper_bounds = [interval[1] for interval in bounds]

        return gym.spaces.Box(low=np.min(lower_bounds),
                              high=np.max(upper_bounds),
                              shape=(n_total_objects, n_dim))

    def generate_batch(self,
                       batch_size=75) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = []
        y_trues = []
        for i in range(0, batch_size):
            state: np.ndarray = self.env.reset()
            inputs.append(state)
            coordinates = np.asarray(
                self.env.get_object_coordinates()).flatten()
            y_trues.append(coordinates)

        batched_obs = model_utils.batch_env_observations(inputs, self.in_space)
        batched_obs = model_utils.scale_space(state=batched_obs,
                                              space=self.in_space)
        batched_obs = model_utils.nd_list_to_torch(batched_obs)[0]
        # for i in range(0, 99):
        #     first = batched_obs[:, 0, 0, i]
        #     print(np.unique(first.cpu().numpy()))
        # print(exit(9))
        y_trues = np.stack(y_trues)
        y_trues = model_utils.scale_space(state=y_trues, space=self.out_space)
        y_trues = torch.tensor(y_trues).to(**settings.ARGS)

        return batched_obs, y_trues
