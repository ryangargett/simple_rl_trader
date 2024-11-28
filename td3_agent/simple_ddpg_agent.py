from environment.environment_base import SimpleTrader
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from typing import Callable
import torch

# Load up the environment
env = SimpleTrader(["CBA.AX", "SYA.AX"])

# Monitor the environment
monitor_env = Monitor(env)

# Environment Checker
check_env(env)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Define the agent
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

# ?????
env.set_render_episodes(False)

# Train the agent
model.learn(total_timesteps=10000, log_interval=10)

# Save the model
model.save("ddpg_trader")

#????
vec_env = model.get_env()

# remove to demonstrate saving and loading
del model

# Load up the model
model = DDPG.load("ddpg_trader")

mean_reward, std_reward = evaluate_policy(model, monitor_env, n_eval_episodes=10)
print(mean_reward, std_reward)

env.unwrapped.render(mode="review")

"""
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    env.render("human")
"""