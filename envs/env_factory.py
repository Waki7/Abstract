from __future__ import annotations

from typing import *

import gym.envs.registration as gym_reg
import gym

if TYPE_CHECKING:
    from envs.training.base_env_encoding_task import EnvEncodingTask

STATE_TRAININGS: Dict[str, Type[EnvEncodingTask]] = {}
IMAGE_ENCODERS = {}
SPEECH_ENCODERS = {}


def get_state_trainer(env_cfg) -> EnvEncodingTask:
    env_id = env_cfg.get('name', None)
    state_encoder = STATE_TRAININGS.get(env_id, None)
    if state_encoder is None:
        raise NotImplementedError(
            'state trainer is not implemented for this environment')
    return state_encoder(env_cfg)


def get_env(env_cfg: Dict) -> gym.Env:
    if len(env_cfg) > 1:
        return gym.make(env_cfg['name'], cfg=env_cfg)
    else:
        return gym.make(env_cfg['name'])


def register(id, entry_point, state_training: EnvEncodingTask = None,
             image_encoder=None, speech_encoder=None):
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
