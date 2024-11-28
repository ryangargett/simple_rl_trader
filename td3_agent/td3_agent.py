from environment.environment_jordan import Trader
import logging

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from typing import Callable

def make_env(rank: int, seed: int = 0):

    def _init():
        env = Trader(ticker_list=['CBA.AX', 'SYA.AX', 'BRN.AX'], initial_funds=2000, observation_metrics=5, starting_date="2023-04-05", ending_date="2023-10-05")
        env.set_render_episodes(False)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return (progress_remaining * (initial_value + final_value)) + final_value
    return func

if __name__ == '__main__':
    logging.basicConfig(filename='Trader.log', encoding='utf-8', level=logging.INFO)

    # Time based hyper-parameters
    n_steps = 1000
    n_epochs = 128

    # Make the environment
    num_cpus = 1
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpus)])

    # Define the model
    model = TD3("MlpPolicy", vec_env, verbose=1, learning_rate=linear_schedule(1e-3, 1e-4))

    # Train the model
    model.learn(total_timesteps=num_cpus*n_steps*n_epochs)

    # Save the model
    model.save("ddpg_trader")

    fun = make_env(0)
    env_1 = fun()
    mon_env = Monitor(env_1)
    obs = env_1.reset()
    env_1.set_render_episodes(True)
    mean_reward, std_reward = evaluate_policy(model, mon_env, n_eval_episodes=12, deterministic=False)
    env_1.unwrapped.render(mode="review")

