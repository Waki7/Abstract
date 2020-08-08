import gym

import grid_world.envs as envs


class EnvEncoderTrainer:
    def __init__(self, env_cfg: dict):
        env: envs.GridEnv = gym.make(env_cfg['name'], cfg=env_cfg)
        self.env = env
        self.env_cfg = env_cfg

    def get_in_spaces(self) -> gym.Space:
        raise NotImplementedError

    def calc_out_space(self) -> gym.Space:
        raise NotImplementedError

    def generate_batch(self):
        self.env.reset()
