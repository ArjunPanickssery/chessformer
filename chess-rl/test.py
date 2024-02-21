import gym
import numpy as np
from stable_baselines3 import DQN

import wandb
from mateenv3 import MateEnv

from stable_baselines3.common.base_class import BaseAlgorithm

env = MateEnv()

wandb.init(
    sync_tensorboard=True,
    project="chessformer",
    save_code=True,
)

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{wandb.run.id}")


# Random Agent, before training
mean_reward_before_train = evaluate(model, num_episodes=100, deterministic=True)

wandb_run = wandb.init(
    project="chessformer",
    name="chess-rl"
)

for i in range(20):
    model.learn(total_timesteps=500_000, log_interval=10_000)
    model.save(f"dqn_chess_{i}")

wandb.finish()

# mean_reward_after_train = evaluate(model, num_episodes=100, deterministic=True)

obs = env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action, render=True)
    # env.render()
