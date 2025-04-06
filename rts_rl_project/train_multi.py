from ray import tune
from ray.rllib.algorithms.ppo import PPO
from multi_rts_env import MultiRTSEnv

config = {
    "env": MultiRTSEnv,
    "num_workers": 2,
    "framework": "torch",
}
tune.run(PPO, config=config, stop={"timesteps_total": 100000})
