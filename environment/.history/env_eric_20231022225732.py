import gymnasium
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from termcolor import colored
import seaborn as sns


class SimpleTrader(gymnasium.Env):

    '''
    CHANGE LOGS:
    - change obseravtion space - altered get_observation and self.state_space_dim function 
    - change reward functio nto be simplier for testing purposes.  
    - change self porfolio value to be the closing price instead of the opening price 
    '''

    # required by gym
    metadata = {"render.modes": ["human", "graph", "review"]}

    def __init__(self, ticker_list, initial_funds=2000, starting_date="2023-07-05", ending_date="2023-10-05", purchase_fee=0.01):
        super().__init__()

        self.starting_date = starting_date
        self.ending_date = ending_date

        self.ticker_list = ticker_list
        self.num_stocks = len(self.ticker_list)

        self.purchase_fee = purchase_fee

        self.stock_data = self._get_stock_data()

        self.state_space_dim = 3 * self.num_stocks + 1

        all_days = pd.date_range(start=self.starting_date, end=self.ending_date, freq="B")

        actual_days = self.stock_data.index.get_level_values("Date").unique()

        self.trading_days = all_days.intersection(actual_days)

        self.num_trading_days = len(self.trading_days)

        print(f"Trading days: {self.num_trading_days}")

        # buy, sell, shares or hold for each stock
        self.action_space = Box(low=-1,
                                high=1,
                                shape=(self.num_stocks,))

        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=(self.state_space_dim,))

        self.initial_funds = initial_funds

        self.episode_funds, self.episode_portfolio = [], []

        self.funds_history, self.portfolio_history, self.rewards = [], [], []

        self.total_buy_actions = []
        self.total_sell_actions = []

        self.avg_reward = 0
        self.reset()

    def step(self, action):

        self.curr_step += 1

        done = self.curr_step >= self.num_trading_days

        reward = self._perform_action(action)

        if not done:
            observation = self._get_observation()
            self.episode_portfolio.append(self.portfolio_value)
            self.episode_funds.append(self.curr_funds)
        else:
            self._get_total_action_count()
            print(f"Bought {self.num_buys} times and Sold {self.num_sells} times")
            observation = None
            self.portfolio_history.append(self.episode_portfolio)
            self.funds_history.append(self.episode_funds)

        return observation, reward, done, False, {}

    def reset(self, seed=1234):

        if seed != None:
            np.random.seed(seed)

        self.episode_portfolio, self.episode_funds, self.rewards = [], [], []

        self.avg_reward = 0

        self.curr_step = 0
        self.curr_funds = self.initial_funds
        self.portfolio_value = self.initial_funds

        self.episode_portfolio.append(self.portfolio_value)

        self.num_buys, self.num_sells = 0, 0
        self.owned_shares = np.zeros(self.num_stocks)

        observation = self._get_observation()

        return observation, {}

    def render(self, mode):
        if mode == "human":
            sns.lineplot(data=self.portfolio_history[-10:])

    def _get_total_action_count(self):
        self.total_buy_actions.append(self.num_buys)
        self.total_sell_actions.append(self.num_sells)

    def _perform_action(self, action_list):

        curr_date = self.trading_days[self.curr_step - 1]

        opening_price = [self.stock_data.loc[(ticker, curr_date), "Open"] for ticker in self.ticker_list]
        closing_price = [self.stock_data.loc[(ticker, curr_date), "Adj Close"] for ticker in self.ticker_list]

        for sell_stock in range(self.num_stocks):

            action = action_list[sell_stock]

            if action < 0:
                max_sell_shares = self.owned_shares[sell_stock]
                num_shares = int(abs(action) * max_sell_shares)
                print(colored(f'SOLD {num_shares} {self.ticker_list[sell_stock]} at {opening_price[sell_stock]}', 'green'))
                if num_shares >= 1:
                    profit = num_shares * opening_price[sell_stock]
                    self.owned_shares[sell_stock] -= num_shares
                    self.curr_funds += profit
                    self.num_sells += 1

        for buy_stock in range(self.num_stocks):

            action = action_list[buy_stock]

            if action > 0:
                max_buy_shares = self.curr_funds / opening_price[buy_stock]
                num_shares = int(action * max_buy_shares)
                print(colored(f'BOUGHT {num_shares} {self.ticker_list[sell_stock]} at {opening_price[sell_stock]}', 'red'))
                if num_shares >= 1:
                    cost = num_shares * (opening_price[buy_stock] + self.purchase_fee)
                    if cost <= self.curr_funds:
                        self.owned_shares[buy_stock] = self.owned_shares[buy_stock] + num_shares
                        self.curr_funds = self.curr_funds - cost
                        self.num_buys = self.num_buys + 1

        self.portfolio_value = self.curr_funds + sum(self.owned_shares * closing_price)

        reward = self._get_reward()

        return reward

    def _get_reward(self):

        reward = self.initial_funds - self.portfolio_value

        return reward
    
    def _get_stock_data(self):

        stock_data_list = []

        for ticker in self.ticker_list:
            ticker_data = yf.download(
                ticker, start=self.starting_date, end=self.ending_date, interval="1d")
            stock_data_list.append(ticker_data)

        stock_data = pd.concat(stock_data_list, keys=self.ticker_list, names=["Ticker", "Date"])

        return stock_data

    def _get_observation(self):

        today = self.trading_days[self.curr_step - 1]
        yesterday = self.trading_days[self.curr_step - 2]
        day_before = self.trading_days[self.curr_step - 3]

        today_price = np.array([self.stock_data.loc[(ticker, today), "Open"] for ticker in self.ticker_list])         # todays price
        yesterday_price = np.array([self.stock_data.loc[(ticker, yesterday), "Open"] for ticker in self.ticker_list]) # yesterdays price
        day_before_price = np.array([self.stock_data.loc[(ticker, day_before), "Open"] for ticker in self.ticker_list]) # day before price

        observation = np.hstack((self.curr_funds, self.owned_shares, today_price, yesterday_price, day_before_price)).astype(np.float32)

        return observation
