import gymnasium as gym
from rts_env import RTSEnv

env = RTSEnv(grid_size=10, num_units=3, max_steps=100)
obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

env.close()
