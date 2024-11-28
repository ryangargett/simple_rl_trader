# Author: Jordan Richards
# Date: 19/10/2023
# Purpose: Provide an alternative abstraction for the properties and computations of 'Shares'
# Todo: None

# Imports
import pandas as pd

class Shares():
    def __init__(self, ticker):
        # Establish the name of the stock/asset
        self.ticker = ticker

        # A list of buy transactions
        self.purchase_list = []
        self.total_bought = 0
        self.num_buys = 0

        # A list of sell transactions
        self.sold_list = []
        self.total_sold = 0
        self.num_sells = 0

        # Total number of shares
        self.num_shares = 0

        # The accumulated buy / sell price
        self.aggregated_buy_price = 0
        self.aggregated_sell_price = 0

        # Profit or loss against the market
        self.historic_positions = [0, 0]
        self.position = 0
        self.holdings_value = 0
        self.market_value = 0


        # Data tracking
        self.historic_closing_price = []
        self.current_closing_price = 0

        self.historic_opening_price = []
        self.current_opening_price = 0


    def buy_shares(self, date, quantity, price):
        share_record = {
            'Date': date,
            'Price': price,
            'Quantity': quantity
        }
        self.purchase_list.append(share_record)
        self.num_buys += 1
        self._get_aggregated_buy_price(share_record)

        self._update()

    def sell_shares(self, date, quantity, price):
        share_record = {
            'Date': date,
            'Price': price,
            'Quantity': quantity
        }
        self.sold_list.append(share_record)
        self.num_sells += 1

        self._update()

    def set_opening_price(self, open_price):
        self.historic_opening_price.append(open_price)
        self.current_opening_price = open_price
        self._update()

    def set_closing_price(self, closing_price):
        self.historic_closing_price.append(closing_price)
        self.current_closing_price = closing_price
        self._update()

    def set_historic_market_prices(self, prices):
        price = list(prices['Open'])
        self.historic_opening_price = price

        price = list(prices['Adj Close'])
        self.historic_closing_price = price

    def _get_num_shares(self):
        self.total_bought = 0
        for record in self.purchase_list:
            self.total_bought += record['Quantity']
        self.total_sold = 0
        for record in self.sold_list:
            self.total_sold += record['Quantity']
        self.num_shares = self.total_bought - self.total_sold

    def _get_aggregated_sell_price(self):
        total = 0
        num = 0
        if self.num_shares != 0:
            for record in self.purchase_list:
                total += record['Price'] * record['Quantity']
                num += record['Quantity']
            self.aggregated_sell_price = total / num
        else:
            self.aggregated_sell_price = 0

    def _get_aggregated_buy_price(self, new_record):
        new_buy = new_record['Price'] * new_record['Quantity']
        added_quantity = new_record['Quantity']
        agg_buy = self.num_shares * self.aggregated_buy_price
        old_num_shares = self.num_shares
        self.aggregated_buy_price = (new_buy + agg_buy) / (added_quantity + old_num_shares)

    def _update_value(self):
        self.holdings_value = self.aggregated_buy_price * self.num_shares

    def _update_market_value(self):
        self.market_value = self.current_closing_price * self.num_shares

    def _profile_or_loss(self):
        self.position = (self.current_closing_price * self.num_shares) - (self.aggregated_buy_price * self.num_shares)
        self.historic_positions.append(self.position)

    def get_historic_closing_df(self):
        nine = pd.DataFrame(self.historic_closing_price[-9:], columns=['Open'])
        twenty = pd.DataFrame(self.historic_closing_price[-21:], columns=['Open'])
        ninety = pd.DataFrame(self.historic_closing_price[-90:], columns=['Open'])
        return nine, twenty, ninety

    def _update(self):
        self._get_num_shares()
        self._update_market_value()
        self._update_value()

    # Output functions
    def print_shares_limited(self):
        print("\n## Stock Summary ##\n")
        print("Stock: ", self.ticker)
        print("Aggregated price per stock: ", self.aggregated_buy_price)
        print("Number of shares: ", self.num_shares)
        print("Profit or loss: ", self.position)
        print("Total value of holdings: ", self.holdings_value)

    def print_buy_transactions(self):
        if self.purchase_list != 0:
            print("\n## Buy Transactions ##\n----------------------")
            for index, record in enumerate(self.purchase_list):
                print("Transaction: " + str(index))
                print("Date: " + str(record['Date']))
                print("Quantity: " + str(record['Quantity']))
                print("Price: " + str(record['Price']))
                print("----------------------")

    def print_sell_transactions(self):
        if self.sold_list != 0:
            print("\n## Sell Transactions ##\n----------------------")
            for index, record in enumerate(self.sold_list):
                print("Transaction: " + str(index))
                print("Date: " + str(record['Date']))
                print("Quantity: " + str(record['Quantity']))
                print("Price: " + str(record['Price']))
                print("----------------------")