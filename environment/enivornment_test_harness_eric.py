import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from environment.environment_eric import EricTrader
import math

CBA=0
SYA=1
BRN=2

def print_data(obs, reward, done, truncated, info):
    print("Observation: ", obs)
    print("Action: ", action)
    print("Reward: ", reward)
    print("Done: ", done)
    print("Info: ", info)
    print("############")

# Starting values
stocks = ['CBA.AX', 'SYA.AX', 'BRN.AX']
initial_funds = 2000
observation_metrics = 3
starting_date="2023-04-05"
ending_date="2023-10-05"

# Goes from the Super --> Child
env = EricTrader(stocks, observation_metrics, initial_funds, starting_date, ending_date)

action = env.action_space.sample()

action[0] = 0.43670973
action[1] = 0.0423286
action[2] = -0.981054
obs, reward, done, truncated, info = env.step(action)

print("\n*** TEST HARNESS ***\n============================================")
########################## ROUND 1

print("\nSTEP 1 TEST")
print("\nTesting Current funds\n--------------------------------------------")
if math.floor(env.curr_funds) == 1159:
    print("passed")
else:
    print(math.floor(env.curr_funds))
    print("failed")

print("\nTesting amount of stock\n--------------------------------------------")
if env.owned_shares[CBA] == 8:
    print("passed")
else:
    print("failed")
if env.owned_shares[SYA] == 262:
    print("passed")
else:
    print("failed")
if env.owned_shares[BRN] == 0:
    print("passed")
else:
    print("failed")

print("\nTesting total portfolio value\n--------------------------------------------")
if math.floor(env.portfolio_value) == 1984:
    print("passed")
else:
    print("failed")
    print(math.floor(env.curr_funds))

########################## ROUND 2

action[0] = -0.7727
action[1] = 0.1271
action[2] = 0.9950643
obs, reward, done, truncated, info = env.step(action)

print("\nSTEP 2 TEST")

print("\nTesting Current funds\n--------------------------------------------")
if math.floor(env.curr_funds) == 8:
    print("passed")
else:
    print("failed")

print("\nTesting amount of stock\n--------------------------------------------")
if env.owned_shares[CBA] == 2:
    print("passed")
else:
    print("failed")
if env.owned_shares[SYA]== 1406:
    print("passed")
else:
    print("failed")
if env.owned_shares[BRN] == 3351:
    print("passed")
else:
    print("failed")

print("\nTesting total portfolio value\n--------------------------------------------")
if math.floor(env.portfolio_value) == 2033:
    print("passed")
else:
    print("failed")
