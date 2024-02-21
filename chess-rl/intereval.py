from stable_baselines3 import DQN

from mateenv3 import MateEnv
from utils import evaluate

env = MateEnv()

for i in range(20):
    # model = DQN("MlpPolicy", env, verbose=1)
    model = DQN.load(f"dqn_chess_{i}")
    model.env = env
    mean_reward = evaluate(model, num_episodes=100, deterministic=True)
    # print(f"Mean reward: {mean_reward:.2f} - Num episodes: {100} - Model: {i}")


# model = DQN("MlpPolicy", env, verbose=1)
# model = DQN.load(f"dqn_chess_12")
#
# obs = env.reset()
# dones = False
# while not dones:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action, render=True)
#     # env.render()
