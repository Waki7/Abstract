import envs.training.maneuver_simple_trainers as training

from envs.env_factory import register

register(
    id='Grid-v0',
    entry_point='grid_world.envs:ManeuverSimple',
    state_training=training.StateEncodingProtocol,
)
