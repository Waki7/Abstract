import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Life-v0',
    entry_point='gym_life.envs:LifeEnv',
)