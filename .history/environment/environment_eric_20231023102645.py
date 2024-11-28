from environment_base import SimpleTrader
import numpy as np

# Eric's implementation of the environment
class EricTrader(SimpleTrader):

    def __init__(self, ticker_list):
        super().__init__(ticker_list, initial_funds=100, starting_date="2023-04-05", ending_date="2023-10-05")

    def reset(self, render=False, seed=None):

        if seed != None:
            np.random.seed(seed)

        if render == True:
            self._render_on_completion()

        if not hasattr(self, "funds_history"):
            self.funds_history, self.portfolio_history = [], []
            self.episode_funds, self.episode_portfolio = [], []
            self.render_episodes = False

        self.curr_step = 0
        self.curr_funds = self.initial_funds
        self.portfolio_value = self.initial_funds

        # self.funds_history.append(self.curr_funds - self.initial_funds)
        # self.portfolio_history.append(self.portfolio_value)

        self.num_buys, self.num_sells = 0, 0
        self.buy_percents, self.sell_percents = 0.0, 0.0
        self.owned_shares = np.zeros(self.num_stocks)

        observation = self._get_observation_eric_0()

        return observation, {}

    def step(self, action_list):

        self.curr_step += 1

        # print(f"Step: {self.curr_step}")

        done = self.curr_step >= self.num_trading_days

        reward = self._perform_action_eric_0(action_list)

        if self.render_episodes == True:
            self.episode_funds.append(self.curr_funds - self.initial_funds)
            self.episode_portfolio.append(self.portfolio_value)

            if done:
                self.funds_history.append(self.episode_funds)
                self.portfolio_history.append(
                    self.episode_portfolio)
                self.episode_funds, self.episode_portfolio = [], []
                self._get_total_action_count()
                self._get_total_buy_sell_percents()

        if not done:
            observation = self._get_observation_eric_0()
        else:
            observation = None

        return observation, reward, done, False, {}

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

        return reward

    def _get_reward_eric_0(self, action_list, buy_reward, sell_reward, money_change, startPortfolio):
        # Reward based on increasing portfolio value from initial value
        if self.curr_step != len(self.trading_days):
            next_date = self.trading_days[self.curr_step]
        else:
            next_date = self.trading_days[self.curr_step - 1]

        next_opening_price = [self.stock_data.loc[(
                                                      ticker, next_date), "Open"] for ticker in self.ticker_list]

        stockVal = 0
        for ii in range(len(self.owned_shares)):
            stockVal += (self.owned_shares[ii] * next_opening_price[ii])
        next_portfolio = self.curr_funds + stockVal

        reward = (100 * (next_portfolio - self.portfolio_value) / next_portfolio)

        curr_date = self.trading_days[self.curr_step - 1]
        # prev_date = self.trading_days[self.curr_step - 2]

        opening_price = [self.stock_data.loc[(ticker, curr_date), "Open"] for ticker in self.ticker_list]

        # Adds reward for buying when a price is going up and selling when a price is going down
        # for ii, action in enumerate(action_list):
        #    reward += (money_change[ii] * action * (next_opening_price[ii] - opening_price[ii])/next_opening_price[ii])/10

        # Penalty for buying or selling when unable to buy or sell
        reward += (buy_reward + sell_reward)

        print(f"Portfolios: {self.portfolio_value} : {next_portfolio}")

        print(f"reward :{reward}\n")

        return reward

    def _get_observation_eric_0(self):

        curr_date = self.trading_days[self.curr_step - 1]
        if self.curr_step != 1:
            prev_date = self.trading_days[self.curr_step - 2]
        else:
            prev_date = self.trading_days[self.curr_step - 1]

        opening_price = np.array([self.stock_data.loc[(
                                                          ticker, curr_date), "Open"] for ticker in self.ticker_list])
        opening_price_prev = np.array([self.stock_data.loc[(
                                                               ticker, prev_date), "Open"] for ticker in
                                       self.ticker_list])

        volume = np.array([self.stock_data.loc[(ticker, curr_date), "Volume"] for ticker in self.ticker_list])

        observation = [round(self.curr_funds, 3)]
        for ii in range(self.num_stocks):
            observation.append(int(self.owned_shares[ii]))
            observation.append(round(opening_price[ii], 3))
            observation.append(round(opening_price_prev[ii], 3))
            observation.append(round(volume[ii], 3))

        print(f"observation: {observation}")

        return observation