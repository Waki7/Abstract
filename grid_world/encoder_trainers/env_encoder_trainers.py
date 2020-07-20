import typing as typ
import grid_world.envs as envs

import gym


class EnvEncoderTrainer:
    def __init__(self, cfg: dict):
        env: envs.GridEnv = gym.make(cfg['name'], cfg=cfg)
        self.env = env

    def get_in_spaces(self) -> gym.Space:
        raise NotImplementedError

    def get_out_spaces(self) -> gym.Space:
        raise NotImplementedError

    def generate_batch(self):
        self.env.reset()
