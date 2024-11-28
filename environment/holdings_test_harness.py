from Shares import Shares
from Portfolio import Portiolio


########################################
print("\n\n ### TESTING PORTFOLIO ### \n")

initial_funds = 2000
starting_date = "2023-04-04"
ending_date = "2023-04-05"
ticker_list = ['CBA.AX', 'SYA.AX', 'BRN.AX']

# Test portfolio
port = Portiolio(initial_funds, starting_date, ending_date, ticker_list)

ticker = "CBA.AX"
price = 20
quantity = 10
date = "2023-10-05"

port.buy(ticker, date, quantity, price)


assert port.shares[CBA].aggregated_buy_price == 20
assert port.shares[CBA].num_shares == 10
assert port.shares[CBA].holdings_value == 200.0

assert port.current_funds == 1800
assert port.current_portfolio_value == 200
assert port.num_share == 10

ticker = "CBA.AX"
price = 20
quantity = 40
date = "2023-10-05"

port.buy(ticker, date, quantity, price)

assert port.shares[CBA].aggregated_buy_price == 20
assert port.shares[CBA].num_shares == 50
assert port.shares[CBA].holdings_value == 1000.0

assert port.current_funds == 1000
assert port.current_portfolio_value == 1000
assert port.num_share == 50

port.shares[CBA].print_shares_limited()