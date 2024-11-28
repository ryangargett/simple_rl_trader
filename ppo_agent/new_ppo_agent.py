import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.policies import obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback
# from tqdm import tqdm
from tqdm.auto import tqdm, trange
import gymnasium
import torch
from typing import Callable
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO, TRPO
from stable_baselines3 import PPO, DDPG, TD3
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from environment.environment_ryan import Trader
#from environment.environment_jordan import Trader
#from environment.environment_julian import Trader 
#from environment.environment_eric import EricTrader


# from env.gym_trader import SimpleTrader


def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = Trader(ticker_list=["CBA.AX", "SYA.AX", "BRN.AX"], observation_metrics=4, initial_funds=2000, starting_date="2023-04-05", ending_date="2023-10-05")
        #env = EricTrader(ticker_list=["CBA.AX", "SYA.AX", "BRN.AX"], observation_metrics=3, initial_funds=2000)
        # env = gymnasium.make("CartPole-v1")
        env.set_render_episodes(False)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return (progress_remaining * (initial_value + final_value)) + final_value

    return func


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, num_cpus):
        super(ProgressCallback, self).__init__()
        self.pbar = tqdm(total=int(total_timesteps/num_cpus), desc="Training")

    def _on_step(self):
        self.pbar.update()
        return True

    def _on_training_end(self):
        self.pbar.close()


if __name__ == '__main__':
    minibatch_size = 128
    n_minibatches = 16
    n_steps = minibatch_size*n_minibatches
    n_steps = 2048
    num_cpus = 8
    n_epochs = 128
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpus)])
    # vec_norm = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
    # vec_norm.set_venv(vec_env)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[400, 300], vf=[400, 300]))
    
    policy_kwargs2 = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[64, 64], qf=[64, 64]))
    
    policy_kwargs3 = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[64, 64, 64], qf=[64, 64, 64]))

    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model = PPO(policy="MlpPolicy", env=vec_norm, n_steps=n_steps, batch_size=minibatch_size, learning_rate=linear_schedule(
    #    1e-3, 1e-4), ent_coef=0.2, vf_coef=0.99, device='cuda', verbose=1, gamma=0.99, gae_lambda=0.99)  # , policy_kwargs=policy_kwargs)
    #model = PPO("MlpPolicy", vec_env, verbose=1,learning_rate=linear_schedule(1e-3, 1e-4))
    model = TRPO(policy="MlpPolicy", env=vec_env, verbose=1, learning_rate=linear_schedule(1e-3, 1e-4), gae_lambda=0.9)
    #model = TD3(policy="MlpPolicy", env=vec_env, verbose=1, learning_rate=linear_schedule(1e-3, 1e-4), action_noise=action_noise, policy_kwargs=policy_kwargs2)
    # model = RecurrentPPO(policy="MlpLstmPolicy", env=vec_env, n_steps=n_steps, batch_size=minibatch_size, learning_rate=linear_schedule(1e-3, 1e-4), ent_coef=0.1, device='cuda', verbose=1, gamma=0.8)

    

    # model = DDPG(policy="MlpPolicy", env=vec_env, verbose=1,
    #             learning_rate=linear_schedule(1e-3, 1e-4), action_noise=action_noise)
    # model = PPO.load("ppo_trader_4", env=vec_env, device='cuda')

    total_timesteps = num_cpus*n_steps*n_epochs
    callback = ProgressCallback(total_timesteps, num_cpus)

    model.learn(total_timesteps, callback)

    # model.save("recurrent_ppo_trader_2")
    # model.save("ppo_trader_jordan_obs0_rew0")
    # model.save("trpo_trader_obs1_rew3")
    model.save("trpo_ryan_rew7_higher_gae")

    # vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpus)])

    # n_epochs = 128
    # n_actions = vec_env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


    #model2 = PPO(policy="MlpPolicy", env=vec_env, verbose=1, learning_rate=linear_schedule(1e-3, 1e-4))
    # model2 = TD3(policy="MlpPolicy", env=vec_env, verbose=1, learning_rate=linear_schedule(1e-3, 1e-4), action_noise=action_noise)
    # total_timesteps = num_cpus*n_steps*n_epochs
    # callback = ProgressCallback(total_timesteps, num_cpus)

    # model.learn(total_timesteps, callback)
    # model2.save("td3_trader_jordan_obs0_rew1_128epochs")

    fun = make_env(0)
    env_1 = fun()
    mon_env = Monitor(env_1)

    obs = env_1.reset()
    env_1.set_render_episodes(True)
    mean_reward, std_reward = evaluate_policy(
        model, mon_env, n_eval_episodes=10, deterministic=False)
    env_1.unwrapped.render(mode="review")

    # fun2 = make_env(0)
    # env_2 = fun2()
    # mon_env2 = Monitor(env_2)

    # obs2 = env_2.reset()
    # env_2.set_render_episodes(True)
    # mean_reward, std_reward = evaluate_policy(
    #     model2, mon_env2, n_eval_episodes=10, deterministic=False)
    # env_2.unwrapped.render(mode="review")