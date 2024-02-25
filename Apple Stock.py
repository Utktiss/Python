#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas  as pd


# In[5]:


df=pd.read_excel("Apple Stock Market.xlsx")


# In[6]:


df


# In[7]:


import pandas as pd

# Assuming your dates are stored in a column called 'Date' in a DataFrame called 'df'
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.strftime('%d-%m-%Y %H:%M:%S')


# In[9]:


df


# In[8]:


import pandas as pd
import plotly.graph_objects as go

# Assuming your dataset is named 'df'
# Make sure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Create traces for each line
trace_close = go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close')
trace_50ma = go.Scatter(x=df['Date'], y=df['50 day MA'], mode='lines', name='50 day MA')
trace_200ma = go.Scatter(x=df['Date'], y=df['200 day MA'], mode='lines', name='200 day MA')

# Create layout
layout = go.Layout(title='Closing Price and Moving Averages',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Price'))

# Create figure and add traces
fig = go.Figure(data=[trace_close, trace_50ma, trace_200ma], layout=layout)

# Show the figure
fig.show()


# In[11]:


pip install pandas plotly


# In[9]:


import pandas as pd

# Assuming your existing DataFrame is named 'df'
# Create new columns 'Budget', 'Signal', and 'Number_of_Stocks'
df['Budget'] = 100000  # Assuming 100,000 USD as the initial budget
df['Signal'] = 0  # Initializing the Signal column with 0 for 'hold'
df['Number_of_Stocks'] = 0  # Initializing the Number_of_Stocks column with 0

# Generate buy and sell signals based on 50-day MA and 200-day MA
df['Signal'][df['50 day MA'] > df['200 day MA']] = 1  # Buy signal
df['Signal'][df['50 day MA'] < df['200 day MA']] = -1  # Sell signal

# Optimize buy and sell decisions based on the budget
for i in range(1, len(df)):
    if df['Signal'][i] == 1:  # Buy signal
        df.at[i, 'Number_of_Stocks'] = df['Budget'][i-1] // df['Close'][i-1]  # Buy stocks
        df.at[i, 'Budget'] = df['Budget'][i-1] - (df['Number_of_Stocks'][i] * df['Close'][i-1])  # Update budget
    elif df['Signal'][i] == -1:  # Sell signal
        df.at[i, 'Budget'] = df['Budget'][i-1] + (df['Number_of_Stocks'][i-1] * df['Close'][i-1])  # Update budget
        df.at[i, 'Number_of_Stocks'] = 0  # Sell all stocks

# Display the updated DataFrame
print(df)


# In[10]:


df


# In[11]:


df.to_excel('stock_trading_results.xlsx', index=False)


# In[17]:


df


# In[16]:


stock_trading_results


# In[17]:


# Assuming your DataFrame is named 'df'
rows_100_to_150 = df.iloc[100:151]

# Display the selected rows
print(rows_100_to_150)


# In[18]:


# Assuming your DataFrame is named 'df'
rows_100_to_150 = df.iloc[201:251]

# Display the selected rows
print(rows_100_to_150)


# In[19]:


rows_100_to_150


# In[1]:


pip install yfinance


# In[14]:


stock_trading_results=pd.read_excel("stock_trading_results.xlsx")


# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50_MA, 200_MA, Corpus, Stocks_Left, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Add a new column 'Portfolio_Value' to track the total value of the portfolio
stock_trading_results['Portfolio_Value'] = stock_trading_results['Corpus'] + (
    stock_trading_results['Stocks_Left'] * stock_trading_results['Close']
)

# Calculate the total profit at the end
initial_budget = stock_trading_results['Budget'].iloc[0]
final_value = stock_trading_results['Portfolio_Value'].iloc[-1]
total_profit = final_value - initial_budget

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50_MA'], label='50-day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200_MA'], label='200-day MA', linestyle='--')

# Highlight Buy signals
plt.scatter(stock_trading_results[stock_trading_results['Signal'] == 1]['Date'],
            stock_trading_results[stock_trading_results['Signal'] == 1]['Close'],
            marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
plt.scatter(stock_trading_results[stock_trading_results['Signal'] == -1]['Date'],
            stock_trading_results[stock_trading_results['Signal'] == -1]['Close'],
            marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Print the total profit at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit: {total_profit}")


# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50_MA, 200_MA, Stocks_Left, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Add a new column 'Portfolio_Value' to track the total value of the portfolio
stock_trading_results['Portfolio_Value'] = stock_trading_results['Stocks_Left'] * stock_trading_results['Close']

# Calculate the total profit at the end
initial_budget = stock_trading_results['Budget'].iloc[0]
final_value = stock_trading_results['Portfolio_Value'].iloc[-1]
total_profit = final_value - initial_budget

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50_MA'], label='50-day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200_MA'], label='200-day MA', linestyle='--')

# Highlight Buy signals
plt.scatter(stock_trading_results[stock_trading_results['Signal'] == 1]['Date'],
            stock_trading_results[stock_trading_results['Signal'] == 1]['Close'],
            marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
plt.scatter(stock_trading_results[stock_trading_results['Signal'] == -1]['Date'],
            stock_trading_results[stock_trading_results['Signal'] == -1]['Close'],
            marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Print the total profit at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit: {total_profit}")


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50_MA, 200_MA, Number_of_Stocks, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Add a new column 'Portfolio_Value' to track the total value of the portfolio
stock_trading_results['Portfolio_Value'] = stock_trading_results['Number_of_Stocks'] * stock_trading_results['Close']

# Calculate the total profit at the end
initial_budget = stock_trading_results['Budget'].iloc[0]
final_value = stock_trading_results['Portfolio_Value'].iloc[-1]
total_profit = final_value - initial_budget

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50_MA'], label='50-day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200_MA'], label='200-day MA', linestyle='--')

# Highlight Buy signals
plt.scatter(stock_trading_results[stock_trading_results['Signal'] == 1]['Date'],
            stock_trading_results[stock_trading_results['Signal'] == 1]['Close'],
            marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
plt.scatter(stock_trading_results[stock_trading_results['Signal'] == -1]['Date'],
            stock_trading_results[stock_trading_results['Signal'] == -1]['Close'],
            marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Print the total profit at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit: {total_profit}")


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50 day MA, 200 day MA, Number_of_Stocks, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Add a new column 'Portfolio_Value' to track the total value of the portfolio
stock_trading_results['Portfolio_Value'] = stock_trading_results['Number_of_Stocks'] * stock_trading_results['Close']

# Calculate the total profit at the end
initial_budget = stock_trading_results['Budget'].iloc[0]
final_value = stock_trading_results['Portfolio_Value'].iloc[-1]
total_profit = final_value - initial_budget

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50 day MA'], label='50 day MA', linestyle='--')  # Change this line
plt.plot(stock_trading_results['Date'], stock_trading_results['200 day MA'], label='200 day MA', linestyle='--')  # Change this line

# Highlight Buy signals
plt.scatter(stock_trading_results[stock_trading_results['Signal'] == 1]['Date'],
            stock_trading_results[stock_trading_results['Signal'] == 1]['Close'],
            marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
plt.scatter(stock_trading_results[stock_trading_results['Signal'] == -1]['Date'],
            stock_trading_results[stock_trading_results['Signal'] == -1]['Close'],
            marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Print the total profit at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit: {total_profit}")


# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50 day MA, 200 day MA, Number_of_Stocks, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Change the strategy: Buy when 50 day MA is less than 200 day MA
stock_trading_results['Signal'] = 0  # Initialize signal column with 0

# Buy Signal: 50-day MA crossing below 200-day MA
stock_trading_results.loc[stock_trading_results['50 day MA'] < stock_trading_results['200 day MA'], 'Signal'] = 1

# Sell Signal: 50-day MA crossing above 200-day MA
stock_trading_results.loc[stock_trading_results['50 day MA'] > stock_trading_results['200 day MA'], 'Signal'] = -1

# Assuming you start with 0 stocks and allocate your budget for buying
for index, row in stock_trading_results.iterrows():
    if row['Signal'] == 1 and row['Budget'] > row['Close']:
        # Buy Stocks
        stocks_bought = row['Budget'] // row['Close']
        stock_trading_results.at[index, 'Number_of_Stocks'] += stocks_bought
        stock_trading_results.at[index, 'Budget'] -= stocks_bought * row['Close']

    elif row['Signal'] == -1 and row['Number_of_Stocks'] > 0:
        # Sell Stocks
        stock_trading_results.at[index, 'Budget'] += row['Number_of_Stocks'] * row['Close']
        stock_trading_results.at[index, 'Number_of_Stocks'] = 0

# Add a new column 'Portfolio_Value' to track the total value of the portfolio
stock_trading_results['Portfolio_Value'] = stock_trading_results['Number_of_Stocks'] * stock_trading_results['Close']

# Calculate the total profit at the end
initial_budget = stock_trading_results['Budget'].iloc[0]
final_value = stock_trading_results['Portfolio_Value'].iloc[-1]
total_profit = final_value - initial_budget

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50 day MA'], label='50 day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200 day MA'], label='200 day MA', linestyle='--')

# Highlight Buy signals
plt.scatter(stock_trading_results[stock_trading_results['Signal'] == 1]['Date'],
            stock_trading_results[stock_trading_results['Signal'] == 1]['Close'],
            marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
plt.scatter(stock_trading_results[stock_trading_results['Signal'] == -1]['Date'],
            stock_trading_results[stock_trading_results['Signal'] == -1]['Close'],
            marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Print the total profit at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit: {total_profit}")


# In[24]:


tickers = ["AAPL", "MSFT", "GOOG"]
data = yf.download(tickers, start="2020-01-01", end="2023-12-31")


# In[25]:


data


# In[23]:


import yfinance as yf


# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50 day MA, 200 day MA, Number_of_Stocks, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Add a new column 'Portfolio_Value' to track the total value of the portfolio
stock_trading_results['Portfolio_Value'] = stock_trading_results['Number_of_Stocks'] * stock_trading_results['Close']

# Calculate the total profit at the end
initial_budget = stock_trading_results['Budget'].iloc[0]
final_value = stock_trading_results['Portfolio_Value'].iloc[-1]
total_profit = final_value - initial_budget

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50 day MA'], label='50 day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200 day MA'], label='200 day MA', linestyle='--')

# Buy Signal: 50-day MA crossing above 200-day MA
buy_signals = stock_trading_results[(stock_trading_results['50 day MA'] > stock_trading_results['200 day MA']) &
                                     (stock_trading_results['50 day MA'].shift(1) <= stock_trading_results['200 day MA'].shift(1))]
plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')

# Sell Signal: 50-day MA crossing below 200-day MA
sell_signals = stock_trading_results[(stock_trading_results['50 day MA'] < stock_trading_results['200 day MA']) &
                                      (stock_trading_results['50 day MA'].shift(1) >= stock_trading_results['200 day MA'].shift(1))]
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Print the total profit at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit: {total_profit}")


# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50 day MA, 200 day MA, Number_of_Stocks, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Add a new column 'Portfolio_Value' to track the total value of the portfolio
stock_trading_results['Portfolio_Value'] = stock_trading_results['Number_of_Stocks'] * stock_trading_results['Close']

# Calculate the total profit at the end
initial_budget = stock_trading_results['Budget'].iloc[0]
final_value = stock_trading_results['Portfolio_Value'].iloc[-1]
total_profit = final_value - initial_budget

# Initialize variables for tracking transactions and budget
transactions = []
remaining_budget = initial_budget

# Iterate through the DataFrame to identify buy and sell signals
for index, row in stock_trading_results.iterrows():
    if row['Signal'] == 1 and row['Budget'] > row['Close']:
        # Buy Stocks
        stocks_bought = row['Budget'] // row['Close']
        remaining_budget -= stocks_bought * row['Close']
        stock_trading_results.at[index, 'Number_of_Stocks'] += stocks_bought
        stock_trading_results.at[index, 'Budget'] -= stocks_bought * row['Close']
        transactions.append(f"Buy: {stocks_bought} stocks at {row['Close']} on {row['Date']}. Remaining Budget: {remaining_budget}")

    elif row['Signal'] == -1 and row['Number_of_Stocks'] > 0:
        # Sell Stocks
        remaining_budget += row['Number_of_Stocks'] * row['Close']
        transactions.append(f"Sell: {row['Number_of_Stocks']} stocks at {row['Close']} on {row['Date']}. Remaining Budget: {remaining_budget}")
        stock_trading_results.at[index, 'Budget'] += row['Number_of_Stocks'] * row['Close']
        stock_trading_results.at[index, 'Number_of_Stocks'] = 0

# Print the transaction details
for transaction in transactions:
    print(transaction)

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50 day MA'], label='50 day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200 day MA'], label='200 day MA', linestyle='--')

# Highlight Buy signals
buy_signals = stock_trading_results[(stock_trading_results['50 day MA'] > stock_trading_results['200 day MA']) &
                                     (stock_trading_results['50 day MA'].shift(1) <= stock_trading_results['200 day MA'].shift(1))]
plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
sell_signals = stock_trading_results[(stock_trading_results['50 day MA'] < stock_trading_results['200 day MA']) &
                                      (stock_trading_results['50 day MA'].shift(1) >= stock_trading_results['200 day MA'].shift(1))]
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Print the total profit at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit: {total_profit}")


# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50 day MA, 200 day MA, Number_of_Stocks, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Create a new DataFrame for buy and sell signals only
buy_sell_signals_df = stock_trading_results[(stock_trading_results['50 day MA'] > stock_trading_results['200 day MA']) |
                                            (stock_trading_results['50 day MA'] < stock_trading_results['200 day MA'])]

# Add a new column 'Action' to label buy and sell signals
buy_sell_signals_df['Action'] = np.where(buy_sell_signals_df['50 day MA'] > buy_sell_signals_df['200 day MA'], 'Buy', 'Sell')

# Display the table of buy and sell signals
print(buy_sell_signals_df[['Date', 'Close', '50 day MA', '200 day MA', 'Action']])

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50 day MA'], label='50 day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200 day MA'], label='200 day MA', linestyle='--')

# Highlight Buy signals
buy_signals = buy_sell_signals_df[buy_sell_signals_df['Action'] == 'Buy']
plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
sell_signals = buy_sell_signals_df[buy_sell_signals_df['Action'] == 'Sell']
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50 day MA, 200 day MA, Number_of_Stocks, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Create a new DataFrame for buy and sell signals only
buy_sell_signals_df = stock_trading_results[(stock_trading_results['50 day MA'] > stock_trading_results['200 day MA']) |
                                            (stock_trading_results['50 day MA'] < stock_trading_results['200 day MA'])]

# Add a new column 'Action' to label buy and sell signals
buy_sell_signals_df['Action'] = np.where(buy_sell_signals_df['50 day MA'] > buy_sell_signals_df['200 day MA'], 'Buy', 'Sell')

# Display the table of buy and sell signals
print(buy_sell_signals_df[['Date', 'Close', '50 day MA', '200 day MA', 'Action']])

# Initialize variables for tracking transactions and budget
transactions = []
remaining_budget = initial_budget
stocks_held = 0

# Iterate through the DataFrame to identify buy and sell signals
for index, row in stock_trading_results.iterrows():
    if row['Signal'] == 1 and row['Budget'] > row['Close']:
        # Buy Stocks
        stocks_bought = row['Budget'] // row['Close']
        remaining_budget -= stocks_bought * row['Close']
        stocks_held += stocks_bought
        transactions.append(f"Buy: {stocks_bought} stocks at {row['Close']} on {row['Date']}. Remaining Budget: {remaining_budget}")

    elif row['Signal'] == -1 and stocks_held > 0:
        # Sell Stocks
        remaining_budget += stocks_held * row['Close']
        transactions.append(f"Sell: {stocks_held} stocks at {row['Close']} on {row['Date']}. Remaining Budget: {remaining_budget}")
        stocks_held = 0

# Calculate profit or loss at the end
final_value = remaining_budget + (stocks_held * stock_trading_results['Close'].iloc[-1])
total_profit_loss = final_value - initial_budget

# Display the transaction details
for transaction in transactions:
    print(transaction)

# Print the total profit or loss at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit/Loss: {total_profit_loss}")

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50 day MA'], label='50 day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200 day MA'], label='200 day MA', linestyle='--')

# Highlight Buy signals
buy_signals = buy_sell_signals_df[buy_sell_signals_df['Action'] == 'Buy']
plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
sell_signals = buy_sell_signals_df[buy_sell_signals_df['Action'] == 'Sell']
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50 day MA, 200 day MA, Number_of_Stocks, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Identify the points where 50-day MA and 200-day MA intersect
cross_points = np.where(np.diff(np.sign(stock_trading_results['50 day MA'] - stock_trading_results['200 day MA'])))[0]

# Create a DataFrame for buy and sell signals at the intersection points
buy_sell_signals_df = stock_trading_results.iloc[cross_points].copy()
buy_sell_signals_df['Action'] = np.where(buy_sell_signals_df['50 day MA'] > buy_sell_signals_df['200 day MA'], 'Buy', 'Sell')

# Display the table of buy and sell signals
print(buy_sell_signals_df[['Date', 'Close', '50 day MA', '200 day MA', 'Action']])

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50 day MA'], label='50 day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200 day MA'], label='200 day MA', linestyle='--')

# Highlight Buy signals
buy_signals = buy_sell_signals_df[buy_sell_signals_df['Action'] == 'Buy']
plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
sell_signals = buy_sell_signals_df[buy_sell_signals_df['Action'] == 'Sell']
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals at Intersection Points')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50 day MA, 200 day MA, Number_of_Stocks, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Identify the points where 50-day MA and 200-day MA intersect after the first 200 days
cross_points = np.where(np.diff(np.sign(stock_trading_results['50 day MA'][200:] - stock_trading_results['200 day MA'][200:])))[0] + 200

# Create a DataFrame for buy and sell signals at the intersection points
buy_sell_signals_df = stock_trading_results.iloc[cross_points].copy()
buy_sell_signals_df['Action'] = np.where(buy_sell_signals_df['50 day MA'] > buy_sell_signals_df['200 day MA'], 'Buy', 'Sell')

# Display the table of buy and sell signals
print(buy_sell_signals_df[['Date', 'Close', '50 day MA', '200 day MA', 'Action']])

# Initialize variables for tracking transactions and budget
transactions = []
remaining_budget = initial_budget
stocks_held = 0

# Iterate through the DataFrame to identify buy and sell signals
for index, row in stock_trading_results.iterrows():
    if index in cross_points:
        # At intersection points, execute buy or sell based on the signal
        if row['50 day MA'] > row['200 day MA'] and row['Budget'] > row['Close']:
            # Buy Stocks
            stocks_bought = row['Budget'] // row['Close']
            remaining_budget -= stocks_bought * row['Close']
            stocks_held += stocks_bought
            transactions.append(f"Buy: {stocks_bought} stocks at {row['Close']} on {row['Date']}. Remaining Budget: {remaining_budget}")

        elif row['50 day MA'] < row['200 day MA'] and stocks_held > 0:
            # Sell Stocks
            remaining_budget += stocks_held * row['Close']
            transactions.append(f"Sell: {stocks_held} stocks at {row['Close']} on {row['Date']}. Remaining Budget: {remaining_budget}")
            stocks_held = 0

# Calculate profit or loss at the end
final_value = remaining_budget + (stocks_held * stock_trading_results['Close'].iloc[-1])
total_profit_loss = final_value - initial_budget

# Display the transaction details
for transaction in transactions:
    print(transaction)

# Print the total profit or loss at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit/Loss: {total_profit_loss}")

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50 day MA'], label='50 day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200 day MA'], label='200 day MA', linestyle='--')

# Highlight Buy signals
buy_signals = buy_sell_signals_df[buy_sell_signals_df['Action'] == 'Buy']
plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
sell_signals = buy_sell_signals_df[buy_sell_signals_df['Action'] == 'Sell']
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

plt.title('AAPL Stock Price with Buy/Sell Signals and Profit/Loss')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_trading_results' with the mentioned columns
# Date, Close, 50 day MA, 200 day MA, Number_of_Stocks, Budget, Signal

# ... (previous code to calculate signals and trading strategy)

# Identify the points where 50-day MA and 200-day MA intersect after the first 200 days
cross_points = np.where(np.diff(np.sign(stock_trading_results['50 day MA'][200:] - stock_trading_results['200 day MA'][200:])))[0] + 200

# Create a DataFrame for buy and sell signals at the intersection points
buy_sell_signals_df = stock_trading_results.iloc[cross_points].copy()
buy_sell_signals_df['Action'] = np.where(buy_sell_signals_df['50 day MA'] > buy_sell_signals_df['200 day MA'], 'Buy', 'Sell')

# Display the table of buy and sell signals
print(buy_sell_signals_df[['Date', 'Close', '50 day MA', '200 day MA', 'Action']])

# Initialize variables for tracking transactions and budget
transactions = []
remaining_budget = initial_budget
stocks_held = 0

# Add a 'Corpus' column to track the invested amount at each point in time
stock_trading_results['Corpus'] = 0

# Iterate through the DataFrame to identify buy and sell signals
for index, row in stock_trading_results.iterrows():
    if index in cross_points:
        # At intersection points, execute buy or sell based on the signal
        if row['50 day MA'] > row['200 day MA'] and row['Budget'] > row['Close']:
            # Buy Stocks
            stocks_bought = row['Budget'] // row['Close']
            remaining_budget -= stocks_bought * row['Close']
            stocks_held += stocks_bought
            transactions.append(f"Buy: {stocks_bought} stocks at {row['Close']} on {row['Date']}. Remaining Budget: {remaining_budget}")
            stock_trading_results.at[index, 'Corpus'] = stocks_bought * row['Close']

        elif row['50 day MA'] < row['200 day MA'] and stocks_held > 0:
            # Sell Stocks
            remaining_budget += stocks_held * row['Close']
            transactions.append(f"Sell: {stocks_held} stocks at {row['Close']} on {row['Date']}. Remaining Budget: {remaining_budget}")
            stocks_held = 0

# Calculate profit or loss at the end
final_value = remaining_budget + (stocks_held * stock_trading_results['Close'].iloc[-1])
total_profit_loss = final_value - initial_budget

# Display the transaction details
for transaction in transactions:
    print(transaction)

# Print the total profit or loss at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit/Loss: {total_profit_loss}")

# Plotting the stock prices along with Buy and Sell signals
plt.figure(figsize=(12, 6))
plt.plot(stock_trading_results['Date'], stock_trading_results['Close'], label='Closing Price', linewidth=2)
plt.plot(stock_trading_results['Date'], stock_trading_results['50 day MA'], label='50 day MA', linestyle='--')
plt.plot(stock_trading_results['Date'], stock_trading_results['200 day MA'], label='200 day MA', linestyle='--')

# Highlight Buy signals
buy_signals = buy_sell_signals_df[buy_sell_signals_df['Action'] == 'Buy']
plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='g', label='Buy Signal')

# Highlight Sell signals
sell_signals = buy_sell_signals_df[buy_sell_signals_df['Action'] == 'Sell']
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='r', label='Sell Signal')

# Display the 'Corpus' line
plt.plot(stock_trading_results['Date'], stock_trading_results['Corpus'], label='Corpus', linestyle='--')

plt.title('AAPL Stock Price with Buy/Sell Signals, Profit/Loss, and Corpus')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[35]:


# Initialize variables for tracking transactions and budget
transactions = []
remaining_budget = initial_budget
stocks_held = 0

# Add a 'Corpus' column to track the invested amount at each point in time
stock_trading_results['Corpus'] = 0

# Iterate through the DataFrame to identify buy and sell signals
for index, row in stock_trading_results.iterrows():
    if index in cross_points:
        # At intersection points, execute buy or sell based on the signal
        if row['50 day MA'] > row['200 day MA'] and row['Budget'] > row['Close']:
            # Buy Stocks
            stocks_bought = row['Budget'] // row['Close']
            remaining_budget -= stocks_bought * row['Close']
            stocks_held += stocks_bought
            transactions.append(f"Buy: {stocks_bought} stocks at {row['Close']} on {row['Date']}. Remaining Budget: {remaining_budget}")
            stock_trading_results.at[index, 'Corpus'] = stocks_held * row['Close']

        elif row['50 day MA'] < row['200 day MA'] and stocks_held > 0:
            # Sell Stocks
            remaining_budget += stocks_held * row['Close']
            transactions.append(f"Sell: {stocks_held} stocks at {row['Close']} on {row['Date']}. Remaining Budget: {remaining_budget}")
            stocks_held = 0

# Calculate profit or loss at the end
final_value = remaining_budget + (stocks_held * stock_trading_results['Close'].iloc[-1])
total_profit_loss = final_value - initial_budget

# Display the transaction details
for transaction in transactions:
    print(transaction)

# Print the total profit or loss at the end
print(f"Initial Budget: {initial_budget}")
print(f"Final Portfolio Value: {final_value}")
print(f"Total Profit/Loss: {total_profit_loss}")


# In[ ]:




