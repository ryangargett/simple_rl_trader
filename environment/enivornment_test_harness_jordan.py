# Author: Jordan Richards
# Date: 19/10/2023
# Purpose: Test the 'Portfolio', 'Shares' and 'environment_jordan' environment with predictable
#          values for two timesteps.
# Todo: None

# Imports
import math
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from environment.environment_jordan import Trader

# Constants
CBA=0
SYA=1
BRN=2

# Starting values
stocks = ['CBA.AX', 'SYA.AX', 'BRN.AX']
initial_funds = 2000
observation_metrics = 3
starting_date="2023-04-05"
ending_date="2023-10-05"

# Goes from the Super --> Child
env = Trader(stocks, initial_funds, observation_metrics, starting_date, ending_date)

# Get the structure of actions
action = env.action_space.sample()

# Overwrite actions to predicable actions
action[0] = 0.43670973
action[1] = 0.0423286
action[2] = -0.981054
obs, reward, done, truncated, info = env.step(action)

# Perform tests
print("\n*** TEST HARNESS ***\n============================================")
########################## ROUND 1

print("\nSTEP 1 TEST")
print("\nTesting Current funds\n--------------------------------------------")
if math.floor(env.portfolio.current_funds) == 1159:
    print("passed")
else:
    print("failed")

print("\nTesting amount of stock\n--------------------------------------------")
if env.portfolio.shares[CBA].num_shares == 8:
    print("passed")
else:
    print("failed")
if env.portfolio.shares[SYA].num_shares == 262:
    print("passed")
else:
    print("failed")
if env.portfolio.shares[BRN].num_shares == 0:
    print("passed")
else:
    print("failed")

print("\nTesting total portfolio value\n--------------------------------------------")
if math.floor(env.portfolio.total_funds) == 1984:
    print("passed")
else:
    print("failed")

########################## ROUND 2

action[0] = -0.7727
action[1] = 0.1271
action[2] = 0.9950643
obs, reward, done, truncated, info = env.step(action)

print("\nSTEP 2 TEST")

print("\nTesting Current funds\n--------------------------------------------")
if math.floor(env.portfolio.current_funds) == 8:
    print("passed")
else:
    print("failed")

print("\nTesting amount of stock\n--------------------------------------------")
if env.portfolio.shares[CBA].num_shares == 2:
    print("passed")
else:
    print("failed")
if env.portfolio.shares[SYA].num_shares == 1406:
    print("passed")
else:
    print("failed")
if env.portfolio.shares[BRN].num_shares == 3351:
    print("passed")
else:
    print("failed")

print("\nTesting total portfolio value\n--------------------------------------------")
if math.floor(env.portfolio.total_funds) == 2033:
    print("passed")
else:
    print("failed")

