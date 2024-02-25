#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import matplotlib.pyplot as plt 
import warnings 
import yfinance as yf
# Ignore all warnings 
warnings.simplefilter("ignore")


# In[6]:


import yfinance as yf

# Define the stock symbol and time period
stock_symbol = "AAPL"
start_date = "2018-01-01"
end_date = "2022-12-31"

# Download daily stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)
print(data.columns)

# Print the downloaded data
print(data.head())



# In[9]:


data1 = data.reset_index() 
stock_data = data1[['Date','Close']]


# In[10]:


stock_data


# In[14]:


# Calculate the moving average and add it to the new DataFrame 
stock_data['STMA'] = stock_data['Close'].rolling(50).mean()

# Calculate the moving average and add it to the new DataFrame
stock_data['LTMA'] = stock_data['Close'].rolling(200).mean()


# In[22]:


def implement_trading_signals(stock_data):
    stock_data['Signal'] = 'Hold'  # Initialize a 'Signal' column with 'Hold'
    stock_data['STMA_Prev'] = stock_data['STMA'].shift(1)    
    stock_data['LTMA_Prev'] = stock_data['LTMA'].shift(1)

    # Counter for crossovers     
    crossover_count = 0
    buy_occurred = False  # Variable to track if a buy signal has occurred
    
    for i in range(1, len(stock_data)):
        if stock_data['STMA'][i] > stock_data['LTMA'][i] and stock_data['STMA_Prev'][i] <= stock_data['LTMA_Prev'][i]:
            # Buy Signal: STMA crosses above LTMA (and was below LTMA one day prior)
            stock_data.at[i, 'Signal'] = 'Buy'
            crossover_count += 1
            buy_occurred = True
        elif stock_data['STMA'][i] < stock_data['LTMA'][i] and stock_data['STMA_Prev'][i] >= stock_data['LTMA_Prev'][i]:
            # Sell Signal: STMA crosses below LTMA (and was above LTMA one day prior)
            if not buy_occurred:
                # If a buy signal has not occurred, set the current sell signal to 'Hold'
                stock_data.at[i, 'Signal'] = 'Hold'
            else:
                # If a buy signal has occurred, set the current sell signal to 'Sell'
                stock_data.at[i, 'Signal'] = 'Sell'
                crossover_count += 1
    
    # Drop the 'STMA_Prev' and 'LTMA_Prev' columns
    stock_data.drop(['STMA_Prev', 'LTMA_Prev'], axis=1, inplace=True)
    
    print("Total Crossovers:", crossover_count)
    
    return stock_data


# In[23]:


def backtest_strategy(stock_data, initial_capital=1000000, sell_multiplier=0.5):
    capital = initial_capital
    position = 0
    current_buy = 0
    results = pd.DataFrame(columns=['Date', 'Close Price', 'SMA', 'LMA', 'Signal', 'Capital in Hand', 'Stocks in Hand', 'Returns(in %)', 'Total Capital'])
    
    for index, row in stock_data.iterrows():
        if row['Signal'] == 'Buy':
            # Buy Signal: Invest all available capital
            current_buy = capital // row['Close']
            capital = capital - (current_buy * row['Close'])
            position += current_buy
            returns = ((capital - initial_capital) + (position * row['Close'])) / initial_capital * 100
            total_returns = capital + (position * row['Close'])
            results = results.append({'Date': row["Date"], 'Close Price': row['Close'], 'SMA': row['STMA'], 'LMA': row['LTMA'], 'Signal': 'Buy', 'Capital in Hand': capital, 'Stocks in Hand': position, 'Returns(in %)': returns, 'Total Capital': total_returns}, ignore_index=True)
        elif row['Signal'] == 'Sell':
            # Sell Signal: Sell only a fraction of the invested stocks
            sold_position = position * sell_multiplier
            capital += sold_position * row['Close']
            position -= sold_position
            returns = ((capital - initial_capital) + (position * row['Close'])) / initial_capital * 100
            total_returns = capital + (position * row['Close'])
            results = results.append({'Date': row["Date"], 'Close Price': row['Close'], 'SMA': row['STMA'], 'LMA': row['LTMA'], 'Signal': 'Sell', 'Capital in Hand': capital, 'Stocks in Hand': position, 'Returns(in %)': returns, 'Total Capital': total_returns}, ignore_index=True)
        
        # Update the 'Position' column in the stock_data DataFrame
        stock_data.at[index, 'Position'] = position
    
    # Calculate the total portfolio value (capital + stock value)
    stock_data['Portfolio'] = capital + (stock_data['Position'] * stock_data['Close'])
    
    # Calculate daily returns
    stock_data['Daily_Return'] = stock_data['Portfolio'].pct_change()
    
    return results, stock_data


# In[28]:


#Implementing the strategy
stock_data = implement_trading_signals(stock_data)
results, stock_data = backtest_strategy(stock_data, initial_capital=1000000, sell_multiplier=0.5)


# In[29]:


# Display tabular results 
print(results)


# In[30]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(stock_data['Close'], label='Stock Price')
plt.plot(stock_data['STMA'], label='Short-Term MA')
plt.plot(stock_data['LTMA'], label='Long-Term MA')

plt.title('Stock Price and Moving Averages with Buy/Sell/Hold Signals')

# Use the index for both x and y values in scatter plot 
buy_signals = stock_data[stock_data['Signal'] == 'Buy']
sell_signals = stock_data[stock_data['Signal'] == 'Sell']
hold_signals = stock_data[stock_data['Signal'] == 'Hold']

# Scatter plot for Buy and Sell signals
plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal')
plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal')

plt.legend()
plt.show()


# In[ ]:




