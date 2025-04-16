from multi_rts_env import MultiRTSEnv

env = MultiRTSEnv(num_agents=2)
observations = env.reset()

while env.agents:
    for agent in env.agents:
        action = env.action_space(agent).sample()
        env.step(action)
    env.render()
env.close()
