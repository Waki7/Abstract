import typing as typ
import grid_world.encoder_trainers as enc_trainers
import grid_world.envs as envs


class StateEncodingProtocol(enc_trainers.EnvEncoderTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.env: envs.ManeuverSimple

    def get_in_spaces(self) -> typ.List[typ.Tuple[float]]:
        return self.env.get_obs_space()

    def get_out_spaces(self) -> typ.List[typ.Tuple[float]]:
        n_landmarks = self.env.n_landmarks
        bounds = self.env.world.bounds

        print(bounds)
        print(exit(9))

    def generate_batch(self):
        self.env.reset()
