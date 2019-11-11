from gym.envs.registration import register

register(
    id='Life-v0',
    entry_point='gym_life.envs:LifeEnv',
)