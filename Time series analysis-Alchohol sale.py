#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


df=pd.read_csv(r"C:\Users\toutk\OneDrive\Desktop\Alcohol_Sales.csv")


# In[9]:


df


# In[7]:


df.index.freq='MS'


# In[20]:


# Assuming you want to rename the second column to 'sales'
df.columns = ['Date', 'sales']


# In[21]:


df.plot(figsize=(12,8))


# In[22]:


df


# In[24]:


df['sales_lastmonth']=df['sales'].shift(1)
df['sales_2monthback']=df['sales'].shift(2)
df['sales_3lastmonthback']=df['sales'].shift(3)


# In[25]:


df


# In[26]:


df=df.dropna()


# In[27]:


df


# In[28]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[29]:


# Define features (X) and target variable (y)
X = df[['sales_lastmonth', 'sales_2monthback', 'sales_3lastmonthback']]
y = df['sales']


# In[30]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


# Initialize and train the Random Forest regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# In[32]:


# Make predictions
y_pred = model.predict(X_test)


# In[33]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[34]:


from sklearn.metrics import r2_score

# Calculate R-squared value
r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)


# In[35]:


y_pred


# In[36]:


X_train


# In[37]:


X_test


# In[ ]:




