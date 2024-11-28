from environment.Shares import Shares

class Portiolio():
    def __init__(self, initial_funds, starting_date, ending_date, ticker_list, historic_prices):
        self.ticker_list = ticker_list
        self.start_date = starting_date
        self.end_date = ending_date
        self.shares = []
        self.num_share = 0
        self.num_buys = []
        self.num_sells = []
        self.current_funds = initial_funds
        self.historic_funds = [initial_funds] # A record of the funds of the portfolio

        for ticker_name in ticker_list:
            self.shares.append(Shares(ticker_name))

        for asset in range(len(historic_prices)):
            self.shares[asset].set_historic_market_prices(historic_prices[asset])

        # Daily returns in the portfolio
        # - Represents the difference in portfolio value between days
        self.daily_return = 0
        self.historic_daily_return = []


        # The total holding values of the stock
        # - Current market holding value of each stock
        self.current_portfolio_value = 0
        self.historic_portfolio_values = [initial_funds] # A record of this metric

        # Total accumulated position of each stock
        # - Difference between what the stock was bought for and current market price
        self.accumulated_position = 0

        # The total funds in the portfolio
        # - Current funds
        # - Current holdings value of the current market price
        self.total_funds = 0

    # Buy and Sell actions
    def buy(self, ticker, date, quantity, price):
        if self.current_funds >= price * quantity and quantity != 0:
            for share in self.shares:
                if share.ticker == ticker:
                    share.buy_shares(date, quantity, price)
                    self.current_funds -= (price * quantity)
                    self.historic_funds.append(self.current_funds)
                    break
        self.update()

    def sell(self, ticker, date, quantity, price):
        for share in self.shares:
            if share.ticker == ticker:
                if share.num_shares >= quantity and quantity != 0:
                    share.sell_shares(date, quantity, price)
                    self.current_funds += (price * quantity)
                    self.historic_funds.append(self.current_funds)
                    break
        self.update()

    # Metric Calculations
    def set_opening_prices(self, opening_prices):
        for index, opening_price in enumerate(opening_prices):
            self.shares[index].set_opening_price(opening_price)

    def set_closing_prices(self, closing_prices):
        for index, closing_price in enumerate(closing_prices):
            self.shares[index].set_closing_price(closing_price)

    def num_shares(self):
        self.num_share = 0
        for index, _ in enumerate(self.shares):
            self.num_share += self.shares[index].num_shares

    def evaluate(self):
        self.current_portfolio_value = 0
        for index, _ in enumerate(self.shares):
            self.current_portfolio_value += self.shares[index].market_value

    def update_total(self):
        self.total_funds = self.current_portfolio_value + self.current_funds
        self.historic_portfolio_values.append(self.total_funds)

    def update_position(self):
        self.accumulated_position = 0
        for index, _ in enumerate(self.shares):
            self.shares[index]._profile_or_loss()
            self.accumulated_position += self.shares[index].position

    def daily_return(self):
        self.daily_return = (self.historic_portfolio_values[-1] - self.historic_portfolio_values[-2]) / self.historic_portfolio_values[-2]
        self.historic_daily_return.append(self.daily_return)

    def update(self):
        self.num_shares()
        self.evaluate()
        self.update_total()

    def _get_num_buys(self):
        self.num_buys = []
        for index, _ in enumerate(self.shares):
             self.num_buys.append(self.shares[index].num_buys)

    def _get_num_sells(self):
        self.num_sells = []
        for index, _ in enumerate(self.shares):
             self.num_sells.append(self.shares[index].num_sells)

    # Output data
    def print_portfolio(self):
        print("Current Funds: ",self.current_funds)
        print("Current Portfolio Value", self.current_portfolio_value)
        print("Accumalated Position: ", self.accumulated_position)
        print("Total Funds", self.total_funds)

    def print_share_portfolio(self):
        for index, _ in enumerate(self.shares):
            self.shares[index].print_shares_limited()

    def print_all_portfolio(self):
        for index, _ in enumerate(self.shares):
            self.shares[index].print_buy_transactions()
            self.shares[index].print_sell_transactions()