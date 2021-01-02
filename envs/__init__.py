import envs.training.grid_encoding_task as training

from envs.env_factory import register

register(
    id='Grid-v0',
    entry_point='envs.grid_world.env_variations.maneuver_simple:ManeuverSimple',
    state_training=training.GridEncodingTask,
)
