# RL-Project
Technion CRML Lab project: Designing Action Primitives for a Simulated Robotic Grasping Agent
 
The project goal is to test the difference in learning rate and success as a result of different action–spaces.
Defines four high-level actions, aimed to reach, grab and lift an object.
Based on Franka Emika Panda robotic arm simulation. Runs on Gym environment, uses PyBullet simulation.
For RL implementation with PPO algorithm, we use Stable Baselines3.

Main.py – load learned policy and runs with visualization

Learn_ppo.py – define environment as defined in panda_env.py, number of envs, initialize MLP policy, assign parameters to the PPO algorithm.

Panda_env.py – define simulation environment
