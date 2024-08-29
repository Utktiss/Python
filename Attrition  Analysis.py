#!/usr/bin/env python
# coding: utf-8

# In[167]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[168]:


df=pd.read_csv(r"C:\Users\toutk\Downloads\WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[169]:


df


# In[170]:


df.isnull().sum()


# In[171]:


# Filter out non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Define the size of the canvas
fig_width = 15
fig_height = 5 * len(numeric_cols) // 2 + 1  # Adjust the height according to the number of numeric columns

# Create the figure with the adjusted size
plt.figure(figsize=(fig_width, fig_height))

# Iterate through each numeric column in the DataFrame
for i, column in enumerate(numeric_cols):
    plt.subplot(len(numeric_cols)//2 + 1, 2, i+1)  # Adjust subplot layout as needed
    plt.boxplot(df[column])
    plt.title(column)

plt.tight_layout()
plt.show()


# In[172]:


df.columns


# In[173]:


# List of columns to remove or consolidate
columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18', 
                   'MonthlyRate', 'DailyRate', 'HourlyRate', 
                   'YearsSinceLastPromotion', 'YearsWithCurrManager',
                   'JobRole', 'EducationField']

# Drop the identified columns from the DataFrame
df.drop(columns=columns_to_drop, inplace=True)



# In[174]:


df


# In[175]:


# Filter the correlation matrix to show only correlations >= 0.5 or <= -0.5
high_correlation = correlation_matrix[(correlation_matrix >= 0.5) | (correlation_matrix <= -0.5)]

# Plotting the heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(high_correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('High Correlation Matrix (>= 0.5 or <= -0.5)')
plt.show()


# In[176]:


# Find variables with correlation >= 0.7
high_correlation_vars = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) >= 0.7:
            colname = correlation_matrix.columns[i]
            if colname in df.columns:  # Check if the column exists in the DataFrame
                high_correlation_vars.add(colname)

# Drop one of the variables with high correlation
df = df.drop(columns=high_correlation_vars, errors='ignore')




# In[177]:


df


# In[178]:


df.columns


# In[179]:


df


# In[180]:


df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['MaritalStatus'] = df['MaritalStatus'].map({'Married': 1, 'Single': 0})
df['BusinessTravel'] = df['BusinessTravel'].map({'Travel_Rarely': 1, 'Travel_Frequently': 0})
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})


# In[156]:


df['Department'].unique()


# In[182]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform 'Department' column
df['Department'] = label_encoder.fit_transform(df['Department'])


# In[185]:


df


# In[186]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Assuming df is your DataFrame with the data
# Splitting the data into features (X) and target variable (y)
X = df.drop(columns=['Attrition'])  # Features
y = df['Attrition']  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# In[187]:


# Access coefficients of independent variables
coefficients = model.coef_

# Access intercept
intercept = model.intercept_

print("Coefficients of independent variables:")
for i, coef in enumerate(coefficients[0]):
    print(f"Feature {X.columns[i]}: {coef}")

print("\nIntercept:", intercept)


# In[ ]:




