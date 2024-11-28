import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

#from env.gym_trader import SimpleTrader
#from environment.environment_julian import Trader
#from environment.environment_jordan import Trader
from environment.environment_ryan import Trader
from environment.environment_eric import EricTrader
from stable_baselines3 import PPO, DDPG, TD3
from sb3_contrib import RecurrentPPO, TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from typing import Callable
import torch
import numpy as np

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = Trader(ticker_list=["CBA.AX", "SYA.AX", "BRN.AX"], initial_funds=2000, observation_metrics=4)#, starting_date="2021-04-05", ending_date="2021-10-05") #Julian & Ryan
        #env = Trader(ticker_list=["CBA.AX", "SYA.AX", "BRN.AX"], initial_funds=2000, observation_metrics=5, starting_date="2023-04-05", ending_date="2023-10-05") #Jordan
        #env = EricTrader(ticker_list=["CBA.AX", "SYA.AX", "BRN.AX"], initial_funds=2000, observation_metrics=3) #Eric
        #env = Trader(ticker_list=["CBA.AX", "ASX.AX", "FMG.AX"], initial_funds=2000)
        #env = Trader(ticker_list=["CBA.AX", "SYA.AX"], initial_funds=2000)
        env.set_render_episodes(False)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # fun = make_env(0)
    # env = fun()
    # mon_env = Monitor(env)

    # model = PPO.load("ppo_trader_obs0_rew3_128epochs")
    # model = TRPO.load("trpo_trader_obs0_rew2_128epochs")

    # obs = env.reset()
    # env.set_render_episodes(True)
    # mean_reward, std_reward = evaluate_policy(
    #     model, mon_env, n_eval_episodes=10, deterministic=False)
    # env.unwrapped.render(mode="review")

    # model = TRPO.load("trpo_trader_obs0_rew3_128epochs")

    # fun = make_env(0)
    # env = fun()
    # mon_env = Monitor(env)

    # obs = env.reset()
    # env.set_render_episodes(True)
    # mean_reward, std_reward = evaluate_policy(
    #     model, mon_env, n_eval_episodes=10, deterministic=False)
    # env.unwrapped.render(mode="review")

    #model = TRPO.load("models/trpo_ryan_rew7_128")
    model = PPO.load("models/ppo_ryan_rew7_128")

    #env = Trader(ticker_list=["CBA.AX", "SYA.AX", "BRN.AX"], initial_funds=2000, observation_metrics=4)

    fun = make_env(0)
    env = fun()
    mon_env = Monitor(env)

    obs = env.reset()
    env.set_render_episodes(True)
    mean_reward, std_reward = evaluate_policy(
        model, mon_env, n_eval_episodes=10, deterministic=False)
    env.unwrapped.render(mode="review")

    # model = PPO.load("trpo_ryan_rew7_128")

    # fun = make_env(0)
    # env = fun()
    # mon_env = Monitor(env)

    # obs = env.reset()
    # env.set_render_episodes(True)
    # mean_reward, std_reward = evaluate_policy(
    #     model, mon_env, n_eval_episodes=10, deterministic=False)
    # env.unwrapped.render(mode="review")

    # fun = make_env(0)
    # env = fun()
    # mon_env = Monitor(env)
    
    # obs = env.reset()
    # current_step = env.curr_step
    # num_steps = env.num_trading_days
    
    # env.set_render_episodes(flag=True)
    # action_list = env.num_stocks
    # action_list = np.random.uniform(-1, 1, size=(action_list,))

    # while current_step < num_steps:
    #     new_state, reward, terminal, info, _ = env.step(action_list)
    #     print(current_step)
    #     action_list = np.random.uniform(-1, 1, size=(len(action_list),))

    #     current_step = env.curr_step
    # env.unwrapped.render(mode="review")