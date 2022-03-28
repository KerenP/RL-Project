from gym.envs.registration import register
from envs.panda_env import PandaEnv
register(
    id='panda-v0',
    entry_point='envs:PandaEnv',
)