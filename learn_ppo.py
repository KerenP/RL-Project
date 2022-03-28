import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import envs

# Parallel environments
env = make_vec_env("panda-v0", n_envs=4)

def l_rate(progress):
    if progress >= 2/3:
        return 3e-4
    elif progress >= 1/3:
        return 7e-5
    else:
        return 3e-5

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_log", learning_rate=l_rate)
callback = EvalCallback(env, best_model_save_path="results34/")
model.learn(total_timesteps=2000000, callback=callback)
model.save("ppo_panda_34")
env.close()