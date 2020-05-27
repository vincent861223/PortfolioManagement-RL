
# from environment.env import StockEnv
# import numpy as np


# if __name__ == '__main__':
#     env = StockEnv("Data");
#     print(env)
#     action = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#     obs, reward, done, _ = env.step(action)
#     print(reward)
#     #env.seed(10)


import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from environment.env import StockEnv
import pandas as pd
# The algorithms require a vectorized environment to run
#env = DummyVecEnv([lambda: StockTradingEnv(df)])
env = StockEnv("Data");
model = PPO2(MlpPolicy, env, verbose=1)
print(env.observation_space.shape)
obs = env.reset()
for i in range(2000):
	model.learn(total_timesteps=1)
	action, _states = model.predict(obs)
	obs, rewards, done, info = env.step(action)
	env.render()