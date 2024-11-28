from environment_base import SimpleTrader
import numpy as np
from termcolor import colored

# Eric's implementation of the environment
class EricTrader(SimpleTrader):

    def __init__(self, ticker_list):
        super().__init__(ticker_list, initial_funds=5000, starting_date="2023-04-05", ending_date="2023-10-05")
    

    def step(self, action_list):

        self.curr_step += 1

        done = self.curr_step >= self.num_trading_days

        reward = self._perform_action_eric_0(action_list)

        if self.render_episodes == True:
            self.episode_funds.append(self.curr_funds - self.initial_funds)
            self.episode_portfolio.append(self.portfolio_value)

            if done:
                self.funds_history.append(self.episode_funds)
                self.portfolio_history.append(self.episode_portfolio)
                self.episode_funds, self.episode_portfolio = [], []
                self._get_total_action_count()
                self._get_total_buy_sell_percents()

        if not done:
            observation = self._get_observation_eric_0()
        else:
            observation = None

        return observation, reward, done, False, {}

    def _perform_action_eric_0(self, action_list):
        
        curr_date = self.trading_days[self.curr_step - 1]

        opening_price = [self.stock_data.loc[(ticker, curr_date), "Open"] for ticker in self.ticker_list]
        closing_price = [self.stock_data.loc[(ticker, curr_date), "Adj Close"] for ticker in self.ticker_list]

        self.previous_portfolio = self.portfolio_value

        for sell_stock in range(self.num_stocks):

            action = action_list[sell_stock]

            if action < 0:
                max_sell_shares = self.owned_shares[sell_stock]
                num_shares = int(abs(action) * max_sell_shares)
                if num_shares >= 1:
                    revenue = num_shares * opening_price[sell_stock]
                    self.owned_shares[sell_stock] -= num_shares
                    self.curr_funds += revenue
                    self.num_sells += 1
                    print(colored(f'SOLD {num_shares} {self.ticker_list[sell_stock]} for {revenue}', 'green'))


        for buy_stock in range(self.num_stocks):

            action = action_list[buy_stock]

            if action > 0:
                max_buy_shares = self.curr_funds / opening_price[buy_stock]
                num_shares = int(action * max_buy_shares)
                if num_shares >= 1:
                    cost = num_shares * opening_price[buy_stock]
                    if cost <= self.curr_funds:
                        self.owned_shares[buy_stock] = self.owned_shares[buy_stock] + num_shares
                        self.curr_funds = self.curr_funds - cost
                        self.num_buys = self.num_buys + 1
                        print(colored(f'BOUGHT {num_shares} {self.ticker_list[buy_stock]} for {cost}', 'red'))


        self.portfolio_value = self.curr_funds + sum(self.owned_shares * closing_price)

        reward = self._get_reward_eric_0()

        return reward

    def _get_reward_eric_0(self):
        
        #penalty for holding funds scale: (dollar value)
        funds_penalty = - (self.curr_funds * 0.05)

        #reward for making profit from initial funds scale: (dollar value)
        dollar_profit_reward = self.portfolio_value - self.initial_funds

        #reward for making profit from previous porfolio value scale: dollar value
        dollar_portfolio_reward = self.portfolio_value - self.previous_portfolio

        #reward/penalty for making more than the initial investment value
        #setting a minimum % it ahs to return (3%) scale: x% value

        percent_portfolio_return = ((self.portfolio_value - self.initial_funds)/self.initial_funds) * 100
        if  percent_portfolio_return > 3:                       #small reward for performing good
            min_reward = 10 * ( percent_portfolio_return - 3) 
        elif  percent_portfolio_return < 3:                     #larger penalisation for bad performance
            min_reward = -(15 * (3 -  percent_portfolio_return))            
        elif  percent_portfolio_return < 0:                        #negative performance = large penalisation
            min_reward = 50 *  percent_portfolio_return

        #has to make a 3% improve from the previous portfolio
        #calculate percentage return scale: x% value
        percent_prev_portfolio_return = ((self.portfolio_value - self.previous_portfolio)/self.previous_portfolio) * 100
        print(colored(f'previous return: {percent_prev_portfolio_return}', 'blue')) 
        print(colored(f'percentage return: {percent_portfolio_return}', 'blue'))

        if percent_prev_portfolio_return > 3:                       #small reward for performing good
            return_reward = 10 * (percent_prev_portfolio_return - 3) 
        elif percent_prev_portfolio_return < 3:                     #larger penalisation for bad performance
            return_reward = -(15 * (3 - percent_prev_portfolio_return))            
        elif percent_prev_portfolio_return < 0:                        #negative performance = large penalisation
            return_reward = 50 * percent_prev_portfolio_return

        reward = funds_penalty + dollar_profit_reward + dollar_portfolio_reward + percent_portfolio_return + percent_prev_portfolio_return 
        
        print(colored(f'funds penalty {funds_penalty}', 'magenta'))
                print(colored(f'funds penalty {funds_penalty}', 'magenta'))

        print(colored(f'portfolio % return {percent_portfolio_return}', 'magenta'))
        print(colored(f'profit reward {dollar_portfolio_reward}', 'magenta'))
        print(colored(f'portfolio reward {portfolio_reward}', 'magenta'))
        print(colored(f'return reward {return_reward}', 'magenta'))

        
        return reward

    def _get_observation_eric_0(self):

        curr_date = self.trading_days[self.curr_step - 1]

        if self.curr_step != 1:
            yesterday = self.trading_days[self.curr_step - 2]
        else:
            yesterday = self.trading_days[self.curr_step - 1]

        opening_price = np.array([self.stock_data.loc[(ticker, curr_date), "Open"] for ticker in self.ticker_list])
        yesterday_price = np.array([self.stock_data.loc[(ticker, yesterday), "Open"] for ticker in self.ticker_list])
        volume = np.array([self.stock_data.loc[(ticker, curr_date), "Volume"] for ticker in self.ticker_list])

        observation = [round(self.curr_funds, 3)]
        for ii in range(self.num_stocks):
            observation.append(int(self.owned_shares[ii]))
            observation.append(round(opening_price[ii], 3))
            observation.append(round(yesterday_price[ii], 3))
            observation.append(round(volume[ii], 3))
            
        return observation
    
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

        self.num_buys, self.num_sells = 0, 0
        self.buy_percents, self.sell_percents = 0.0, 0.0
        self.owned_shares = np.zeros(self.num_stocks)

        observation = self._get_observation_eric_0()

        return observation, {}