#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# define the time period for the data
end_date = date.today().strftime("%Y-%m-%d")
start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")

# list of stock tickers to download
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']

data = yf.download(tickers, start=start_date, end=end_date, progress=False)
 
    
data



# In[23]:


# reset index to bring Date into the columns for the melt function
data = data.reset_index()



# In[24]:


data


# In[25]:


# melt the DataFrame to make it long format where each row is a unique combination of Date, Ticker, and attributes
data_melted = data.melt(id_vars=['Date'], var_name=['Attribute', 'Ticker'])



# In[26]:


data_melted


# In[27]:


# pivot the melted DataFrame to have the attributes (Open, High, Low, etc.) as columns
data_pivoted = data_melted.pivot_table(index=['Date', 'Ticker'], columns='Attribute', values='value', aggfunc='first')



# In[28]:


data_pivoted


# In[29]:


# reset index to turn multi-index into columns
stock_data = data_pivoted.reset_index()



# In[30]:


print(stock_data.head())


# In[ ]:





# # Now, let’s have a look at the stock market performance of these companies in the stock market over time:

# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns

stock_data['Date'] = pd.to_datetime(stock_data['Date'])



# 
# The line stock_data['Date'] = pd.to_datetime(stock_data['Date'])
# converts the 'Date' column in the stock_data DataFrame to a datetime format.
# This step is important because the 'Date' column might originally be in string 
# format (e.g., "2023-08-26"), which can make it harder to perform operations like
# sorting, filtering by date, or plotting.
# 
# Sorting:
# When dates are in string format, sorting is done lexicographically (i.e., alphabetically). For example, the string "2023-08-26" will be sorted after "2023-08-19" because of the alphabetical order of characters, not the chronological order of dates. This can lead to incorrect sorting results.
# 
# Filtering:
# If dates are strings, comparing them for filtering can be challenging. For instance, filtering for dates greater than "2023-08-20" may not work as expected because string comparisons don't follow chronological rules. Dates like "2023-08-09" might be considered greater than "2023-08-20" due to string comparison.
# 
# Plotting:
# Plotting tools often require dates to be in datetime format to properly handle the x-axis for time series data. If dates are in string format, the plot may not recognize them as dates, leading to improper scaling or axis labeling.

# In[32]:


stock_data['Date']


# In[33]:


stock_data.set_index('Date', inplace=True)
stock_data.reset_index(inplace=True)
plt.figure(figsize=(14, 7))
sns.set(style='whitegrid')

sns.lineplot(data=stock_data, x='Date', y='Adj Close', hue='Ticker', marker='o')

plt.title('Adjusted Close Price Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Adjusted Close Price', fontsize=14)
plt.legend(title='Ticker', title_fontsize='13', fontsize='11')
plt.grid(True)

plt.xticks(rotation=45)

plt.show()


# The graph displays the adjusted close prices of four stocks
# (HDFCBANK.NS, INFY.NS, RELIANCE.NS, TCS.NS) over time from 
# July 2023 to July 2024. It highlights that TCS has the highest 
# adjusted close prices, followed by RELIANCE, INFY (Infosys), and
# HDFCBANK. The prices for RELIANCE and TCS show noticeable upward 
# trends, which indicates strong performance, while HDFCBANK and INFY 
# exhibit more stability with relatively lower price fluctuations.

# In[ ]:





# 
# # Now, let’s compute the 50-day and 200-day moving averages and plot these along with the Adjusted Close price for each stock:

# In[34]:


short_window = 50
long_window = 200

stock_data.set_index('Date', inplace=True)
unique_tickers = stock_data['Ticker'].unique()

for ticker in unique_tickers:
    ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
    ticker_data['50_MA'] = ticker_data['Adj Close'].rolling(window=short_window).mean()
    ticker_data['200_MA'] = ticker_data['Adj Close'].rolling(window=long_window).mean()

    plt.figure(figsize=(14, 7))
    plt.plot(ticker_data.index, ticker_data['Adj Close'], label='Adj Close')
    plt.plot(ticker_data.index, ticker_data['50_MA'], label='50-Day MA')
    plt.plot(ticker_data.index, ticker_data['200_MA'], label='200-Day MA')
    plt.title(f'{ticker} - Adjusted Close and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.bar(ticker_data.index, ticker_data['Volume'], label='Volume', color='orange')
    plt.title(f'{ticker} - Volume Traded')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# For HDFCBANK and INFY, the prices initially decline but later show signs of recovery, as indicated by the moving averages. RELIANCE and TCS display a more consistent upward trend in their adjusted close prices. The volume traded graphs highlight significant trading activity at various points, with spikes indicating high trading volumes, particularly noticeable in HDFCBANK and RELIANCE around early 2024. These insights are crucial for understanding price movements and trading behaviours, which assist in making informed investment decisions.

# In[ ]:





# # Now, let’s have a look at the distribution of daily returns of these stocks:

# In[35]:


stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].pct_change()

plt.figure(figsize=(14, 7))
sns.set(style='whitegrid')

for ticker in unique_tickers:
    ticker_data = stock_data[stock_data['Ticker'] == ticker]
    sns.histplot(ticker_data['Daily Return'].dropna(), bins=50, kde=True, label=ticker, alpha=0.5)

plt.title('Distribution of Daily Returns', fontsize=16)
plt.xlabel('Daily Return', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(title='Ticker', title_fontsize='13', fontsize='11')
plt.grid(True)
plt.tight_layout()
plt.show()


# The distributions are approximately normal, centred around zero, which indicates that most daily returns are close to the average return. However, there are tails on both sides, which reflect occasional significant gains or losses. INFY and RELIANCE appear to have slightly wider distributions, which suggests higher volatility compared to HDFCBANK and TCS.

# In[ ]:





# In[ ]:





# In[36]:


daily_returns = stock_data.pivot_table(index='Date', columns='Ticker', values='Daily Return')
correlation_matrix = daily_returns.corr()

plt.figure(figsize=(12, 10))
sns.set(style='whitegrid')

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f', annot_kws={"size": 10})
plt.title('Correlation Matrix of Daily Returns', fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# INFY and TCS have a high positive correlation (0.72), which indicates that they tend to move in the same direction. HDFCBANK has a moderate positive correlation with RELIANCE (0.32) and a low correlation with INFY (0.17) and TCS (0.10). RELIANCE shows a low correlation with INFY (0.19) and TCS (0.21). These varying correlations suggest potential diversification benefits; combining stocks with lower correlations can reduce overall portfolio risk.

# In[ ]:





# In[ ]:





# # Portfolio Optimization
# 
# Now, using Modern Portfolio Theory, we can construct an efficient portfolio by balancing risk and return. We will:
# 
# Calculate the expected returns and volatility for each stock.
# Generate a series of random portfolios to identify the efficient frontier.
# Optimize the portfolio to maximize the Sharpe ratio, which is a measure of risk-adjusted return.
# Let’s calculate the expected returns and volatility for each stock:

# In[37]:


import numpy as np

expected_returns = daily_returns.mean() * 252  # annualize the returns
volatility = daily_returns.std() * np.sqrt(252)  # annualize the volatility

stock_stats = pd.DataFrame({
    'Expected Return': expected_returns,
    'Volatility': volatility
})

stock_stats


# RELIANCE has the second highest expected return (24.06%) and moderate volatility (21.27%), which indicates a potentially high-reward investment with relatively higher risk. INFY and TCS also have high expected returns (33.03% and 33.13% respectively) with moderate volatility (22.11% and 20.80%). HDFCBANK has the lowest expected return (.07%) and moderate volatility (21.80%), which makes it the least attractive in terms of risk-adjusted returns.

# In[ ]:





# In[ ]:


#Generate a large number of random portfolio weights.
#Calculate the expected return and volatility for each portfolio.
#Plot these portfolios to visualize the efficient frontier.
#Let’s generate the random portfolios and plot the efficient frontier:


# In[38]:


# function to calculate portfolio performance
def portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# number of portfolios to simulate
num_portfolios = 10000

# arrays to store the results
results = np.zeros((3, num_portfolios))

# annualized covariance matrix
cov_matrix = daily_returns.cov() * 252

np.random.seed(42)

for i in range(num_portfolios):
    weights = np.random.random(len(unique_tickers))
    weights /= np.sum(weights)

    portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)

    results[0,i] = portfolio_return
    results[1,i] = portfolio_volatility
    results[2,i] = portfolio_return / portfolio_volatility  # Sharpe Ratio

plt.figure(figsize=(10, 7))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)
plt.show()


# Each dot represents a portfolio, with the colour indicating the Sharpe ratio, a measure of risk-adjusted return. Portfolios on the leftmost edge of the frontier (closer to the y-axis) offer the highest expected returns for a given level of volatility, which represent optimal portfolios. The gradient shows that portfolios with higher Sharpe ratios (darker blue) provide better risk-adjusted returns.

# In[ ]:





# In[39]:


max_sharpe_idx = np.argmax(results[2])
max_sharpe_return = results[0, max_sharpe_idx]
max_sharpe_volatility = results[1, max_sharpe_idx]
max_sharpe_ratio = results[2, max_sharpe_idx]

max_sharpe_return, max_sharpe_volatility, max_sharpe_ratio


# In[40]:


#Next, let’s identify the weights of the stocks in the portfolio that yield the maximum Sharpe ratio:

max_sharpe_weights = np.zeros(len(unique_tickers))

for i in range(num_portfolios):
    weights = np.random.random(len(unique_tickers))
    weights /= np.sum(weights)

    portfolio_return, portfolio_volatility = portfolio_performance(weights, expected_returns, cov_matrix)

    if results[2, i] == max_sharpe_ratio:
        max_sharpe_weights = weights
        break

portfolio_weights_df = pd.DataFrame({
    'Ticker': unique_tickers,
    'Weight': max_sharpe_weights
})

portfolio_weights_df


# TCS has the highest allocation, which indicates its significant contribution to the portfolio’s performance, while INFY has the smallest allocation. This balanced allocation aims to maximize returns while minimizing risk by leveraging individual stock performances and their correlations.

# So, this is how stock market portfolio optimization works. Stock market portfolio optimization involves analyzing price trends, calculating expected returns and volatilities, and determining the correlations between different stocks to achieve diversification.

# In[ ]:




