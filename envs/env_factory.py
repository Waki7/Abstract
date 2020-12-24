from typing import *

import gym.envs.registration as gym_reg

if TYPE_CHECKING:
    from envs.training.env_encoder_trainers import EnvEncoderTrainer

STATE_TRAININGS: Dict[str, Type[EnvEncoderTrainer]] = {}
IMAGE_ENCODERS = {}
SPEECH_ENCODERS = {}


def get_state_trainer(env_cfg) -> EnvEncoderTrainer:
    env_id = env_cfg.get('name', None)
    state_encoder = STATE_TRAININGS.get(env_id, None)
    if state_encoder is None:
        raise NotImplementedError('state trainer is not implemented for this environment')
    return state_encoder(env_cfg)


def register(id, entry_point, state_training: EnvEncoderTrainer = None, image_encoder=None, speech_encoder=None):
    gym_reg.register(
        id=id,
        entry_point=entry_point,
    )
    if state_training is not None:
        STATE_TRAININGS[id] = state_training
    if image_encoder is not None:
        IMAGE_ENCODERS[id] = image_encoder
    if speech_encoder is not None:
        SPEECH_ENCODERS[id] = speech_encoder
