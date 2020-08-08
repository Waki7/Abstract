import gym.envs.registration as gym_reg

STATE_ENCODERS = {}
IMAGE_ENCODERS = {}
SPEECH_ENCODERS = {}


def get_state_trainer(env_cfg):
    env_id = env_cfg.get('name', None)
    state_encoder = STATE_ENCODERS.get(env_id, None)
    if state_encoder is None:
        raise NotImplementedError('state trainer is not implemented for this environment')
    return state_encoder(env_cfg)


def register(id, entry_point, state_trainer=None, image_encoder=None, speech_encoder=None):
    gym_reg.register(
        id=id,
        entry_point=entry_point,
    )
    if state_trainer is not None:
        STATE_ENCODERS[id] = state_trainer
    if image_encoder is not None:
        IMAGE_ENCODERS[id] = image_encoder
    if speech_encoder is not None:
        SPEECH_ENCODERS[id] = speech_encoder
