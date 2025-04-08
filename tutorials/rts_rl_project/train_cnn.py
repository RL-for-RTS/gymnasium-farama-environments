from stable_baselines3 import PPO
from rts_env import RTSEnv

env = RTSEnv()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")
model.learn(total_timesteps=100000, progress_bar=True)
model.save("rts_ppo_mlp")

# Test the trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
