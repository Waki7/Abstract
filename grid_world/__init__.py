from grid_world.env_factory import register
import grid_world.encoder_trainers as trainers

register(
    id='Grid-v0',
    entry_point='grid_world.envs:ManeuverSimple',
    state_trainer=trainers.StateEncodingProtocol,
)


