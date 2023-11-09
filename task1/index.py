import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('tradelog.csv')

# Calculate parameters
total_trades = len(data)
profitable_trades = len(data[data['Exit Price'] > data['Entry Price']])
loss_making_trades = len(data[data['Exit Price'] < data['Entry Price']])
win_rate = profitable_trades / total_trades

data['P&L'] = data['Exit Price'] - data['Entry Price']
average_profit_per_trade = data[data['P&L'] > 0]['P&L'].mean()
average_loss_per_trade = -data[data['P&L'] < 0]['P&L'].mean()  # considering loss as positive value

risk_reward_ratio = abs(average_profit_per_trade / average_loss_per_trade)

loss_rate = 1 - win_rate
expectancy = (win_rate * average_profit_per_trade) - (loss_rate * average_loss_per_trade)

average_ror_per_trade = (expectancy - 0.05) / data['P&L'].std()

rate_of_return = data['P&L'].mean()
risk_free_rate = 0.05
sharpe_ratio = (rate_of_return - risk_free_rate) / data['P&L'].std()

cumulative_returns = np.cumsum(data['P&L'])
peak = cumulative_returns.cummax()
drawdown = cumulative_returns - peak
max_drawdown = drawdown.min()
max_drawdown_percentage = (max_drawdown / peak.max()) * 100

beginning_value = 6500
ending_value = beginning_value + data['P&L'].sum()
num_periods = len(data)
CAGR = (ending_value / beginning_value) ** (1 / num_periods) - 1

calmar_ratio = -CAGR / (max_drawdown_percentage / 100)

# Create a DataFrame for the results
results = pd.DataFrame({
    'Total Trades': [total_trades],
    'Profitable Trades': [profitable_trades],
    'Loss-Making Trades': [loss_making_trades],
    'Win Rate': [win_rate],
    'Average Profit per trade': [average_profit_per_trade],
    'Average Loss per trade': [average_loss_per_trade],
    'Risk Reward ratio': [risk_reward_ratio],
    'Expectancy': [expectancy],
    'Average ROR per trade': [average_ror_per_trade],
    'Sharpe Ratio': [sharpe_ratio],
    'Max Drawdown': [max_drawdown],
    'Max Drawdown Percentage': [max_drawdown_percentage],
    'CAGR': [CAGR],
    'Calmar Ratio': [calmar_ratio]
})

# Save the results to a CSV file
results.to_csv('results.csv', index=False)
