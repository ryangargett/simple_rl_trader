from environment_base import SimpleTrader
from environment_eric import EricTrader

np.set

stocks = ['CBA.AX', 'SYA.AX', 'BRN.AX']

initial_funds = 2000

#env = SimpleTrader(stocks, initial_funds)
env = EricTrader(stocks)

num_steps = 5

for ii in range(num_steps):
    action = env.action_space.sample()

    obs, reward, done, truncated, info = env.step(action)

    print("Action: ", action)
    print("Observation: ", obs)
    print("Reward: ", reward)
    print("Done: ", done)
    print("Info: ", info)

    #env.render(mode="human")

    if done:
        env.reset(render=True)