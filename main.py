import gym
from stable_baselines3 import PPO
import envs
env = gym.make('panda-v0', gui=True)

# Load saved model, reset environment
model = PPO.load("results32/best_model")
obs = env.reset()

# Run graphics of saved model
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
