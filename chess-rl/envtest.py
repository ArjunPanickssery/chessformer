from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import numpy as np

from mateenv3 import MateEnv

env = MateEnv()

model = DQN("MlpPolicy", env, verbose=1)

obs = env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    # action = int(input())
    obs, rewards, dones, info = env.step(action, render=True)
    # env.render()
