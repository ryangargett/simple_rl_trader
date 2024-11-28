from env.gym_trader import SimpleTrader
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

env = SimpleTrader(["CBA.AX", "SYA.AX"])

mon_env = Monitor(env)

check_env(env)

model = PPO("MlpPolicy", mon_env, n_steps=2048,
            learning_rate=1e-3, ent_coef=0.1)
env.set_render_episodes(False)
model.learn(total_timesteps=10000)
env.set_render_episodes(True)
mean_reward, std_reward = evaluate_policy(
    model, mon_env, n_eval_episodes=10)
print(mean_reward, std_reward)

env.unwrapped.render(mode="review")
