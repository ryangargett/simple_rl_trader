from environment.environment_eric import EricTrader
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from typing import Callable
import torch
import gymnasium
from tqdm import tqdm
from stable_baselines3.common.policies import obs_as_tensor

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = EricTrader(ticker_list=["CBA.AX", "SYA.AX"], initial_funds=2000)
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

if __name__ == '__main__':
    minibatch_size = 64
    n_minibatches = 32
    n_steps = minibatch_size*n_minibatches
    num_cpus = 8
    n_epochs = 16 #128
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpus)])
    #vec_norm = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
    #vec_norm.set_venv(vec_env)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[32, 32], vf=[32, 32]))


    #model = PPO(policy="MlpPolicy", env=vec_norm, n_steps=n_steps, batch_size=minibatch_size, learning_rate=linear_schedule(1e-3, 1e-4), ent_coef=0.2, vf_coef=0.99, device='cuda', verbose=1, gamma=0.99, gae_lambda=0.99)#, policy_kwargs=policy_kwargs)
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=linear_schedule(1e-3, 1e-4))
    #model = RecurrentPPO(policy="MlpLstmPolicy", env=vec_env, n_steps=n_steps, batch_size=minibatch_size, learning_rate=linear_schedule(1e-3, 1e-4), ent_coef=0.1, device='cuda', verbose=1, gamma=0.8)
    #model = PPO.load("ppo_trader_4", env=vec_env, device='cuda')
    model.learn(total_timesteps=num_cpus*n_steps*n_epochs)

    #model.save("recurrent_ppo_trader_2")
    model.save("ppo_trader")

    fun = make_env(0)
    env_1 = fun()
    mon_env = Monitor(env_1)

    obs = env_1.reset()
    env_1.set_render_episodes(True)
    mean_reward, std_reward = evaluate_policy(
        model, mon_env, n_eval_episodes=10, deterministic=False)
    env_1.unwrapped.render(mode="review")
