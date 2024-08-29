#!/usr/bin/env python
# coding: utf-8

# In[5]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[6]:


df=pd.read_csv(r"C:\Users\toutk\Downloads\HR.csv")


# In[7]:


df


# In[8]:


df.info()


# In[11]:


df['sales'].unique()


# In[12]:


df['salary'].unique()


# In[13]:


# Define the continuous features
continuous_features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']

# Identify the features to be converted to object data type
features_to_convert = [feature for feature in df.columns if feature not in continuous_features]

# Convert the identified features to object data type
df[features_to_convert] = df[features_to_convert].astype('object')

df.dtypes

df.describe()
# In[17]:


df.describe(include='object')


# In[18]:


# Create subplots for kde plots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

for ax, col in zip(axes.flatten(), continuous_features):
    sns.kdeplot(data=df, x=col, fill=True, linewidth=2, hue='left', ax=ax, palette = {0: '#009c05', 1: 'darkorange'})
    ax.set_title(f'{col} vs Target')

axes[2,1].axis('off')
plt.suptitle('Distribution of Continuous Features by Target', fontsize=22)
plt.tight_layout()
plt.show()


# 
# #Inference:
# #The satisfaction_level plot shows that employees who left the company generally had a lower satisfaction level compared to those who stayed.
# #The last_evaluation plot does not show a clear distinction between the two classes.
# #The number_project plot shows that employees who left the company generally worked on a very high or very low number of projects.
# #The average_montly_hours plot shows that employees who left the company generally worked very long or very short hours.
# #The time_spend_company plot shows that employees who left the company generally spent a moderate amount of time at the company.
# 

# In[ ]:


# List of categorical features
cat_features = ['Work_accident', 'promotion_last_5years', 'sales', 'salary']

# Initialize the plot
fig, axes = plt.subplots(2, 2, figsize=(20, 8))

# Plot each feature
for i, ax in enumerate(axes.flatten()):
    sns.countplot(x=cat_features[i], hue='left', data=df, ax=ax, palette={0: '#009c05', 1: 'darkorange'})
    ax.set_title(cat_features[i])
    ax.set_ylabel('Count')
    ax.set_xlabel('')
    ax.legend(title='Left', loc='upper right')

plt.suptitle('Distribution of Categorical Features by Target', fontsize=22)
plt.tight_layout()
plt.show()


# #Inference:
# #The Work_accident plot shows that employees who had a work accident are less likely to leave the company.
# #The promotion_last_5years plot shows that employees who have not received a promotion in the last 5 years are more likely to leave the company.
# #The sales plot shows the distribution of employees who left or did not leave the company across different departments. It seems that the sales, technical, and support departments have the highest number of employees who left the company.
# #The salary plot shows that employees with a low salary are more likely to leave the company than those with a medium or high salary.

# # Data preprocessing is a crucial step in any machine learning project. 
# #It involves cleaning and transforming raw data into a format that can be understood 
# #by machine learning algorithms. For this project, the following preprocessing steps will be performed:
# 
# #Check Missing Values
# 
# #Categorical Features Encoding
# 
# #Split the Dataset

# '''
# We can make the following decisions:
# 
# Numerical Variables: These are variables that are already in numerical format and do not need encoding.
#                     satisfaction_level: This is a continuous variable ranging from 0 to 1.
#                     last_evaluation: This is a continuous variable ranging from 0 to 1.
#                     number_project: This is a discrete variable representing the number of projects completed by an employee.
#                     average_montly_hours: This is a continuous variable representing the average monthly hours worked by an employee.
#                     time_spend_company: This is a discrete variable representing the number of years an employee has spent at the company.
#     
# Ordinal Variables: These variables have an inherent order. They don't necessarily need to be one-hot 
#                    encoded since their order can provide meaningful information to the model:
#                    salary: This variable has 3 unique values ('low', 'medium', 'high') which have an inherent order.
#                    Therefore, it can be label encoded.
#         
# Nominal Variables: These are variables with no inherent order. They should be one-hot encoded because using 
#                    them as numbers might introduce an unintended order to the model:
#                    sales: This variable represents the department of the employee and has 10 unique values.
#                    It should be one-hot encoded.
#         
# Binary Variables: These are variables with only two categories and do not need to be one-hot encoded:
# 
#                   Work_accident: This is a binary variable (0 or 1).
#                   left: This is the target variable and is binary (0 or 1).
#                 promotion_last_5years: This is a binary variable (0 or 1).
# Summary:
# Need Label Encoding: salary
# Need One-Hot Encoding: sales   
# '''

# In[24]:


#Implementing one-hot encoding on the 'sales' feature
df_encoded = pd.get_dummies(df, columns=['sales'], drop_first=True)

# Label encoding of 'salary' feature
le = LabelEncoder()
df_encoded['salary'] = le.fit_transform(df_encoded['salary'])

df_encoded.head()


# In[40]:


df_encoded.info()


# In[39]:


# Assuming df is your DataFrame
# Convert object-type columns to integers
df_encoded['Work_accident'] = df_encoded['Work_accident'].astype(int)
df_encoded['left'] = df_encoded['left'].astype(int)
df_encoded['promotion_last_5years'] = df_encoded['promotion_last_5years'].astype(int)


# In[41]:


# Define the features (X) and the output labels (y)
X = df_encoded.drop('left', axis=1)
y = df_encoded['left']


# In[42]:


# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


# In[43]:


#Note:
#In the above split, we used Stratification which ensures that the 
#distribution of the target variable (left) is the same in both the train and test sets


# '''
# We are going to use regression algorithms instead of 
# classification algorithms in this project for three main reasons:
# 
# Probabilistic Interpretation: Regression algorithms can predict a 
# continuous output which can be interpreted as the probability of a certain event.
# In this case, the output can be interpreted as the probability that an employee 
# will leave the company. This information can be more informative than just a binary
# output and can help in understanding how 'at risk' each employee is of leaving.
# 
# 
# Threshold Calibration: By predicting probabilities, we can adjust the threshold for 
# classifying an observation as 0 or 1. For example, we might classify all employees 
# with a predicted probability of leaving greater than 0.5 as 'will leave'. However,
# we can adjust this threshold to be more conservative or more liberal depending on
# the cost of false positives and false negatives. For example, if it is more costly 
# to incorrectly predict that an employee will stay when they actually leave, we might
# lower the threshold to 0.3 to identify more employees at risk of leaving.
# 
# 
# Imbalanced Data:  The target variable, left, is imbalanced with a larger
# proportion of employees who did not leave the company. This can sometimes 
# lead to poor performance for classification algorithms because they have a
# bias towards the majority class. Regression algorithms do not have this bias
# and can sometimes perform better on imbalanced data.
# 
# Using regression algorithms will allow for a more nuanced understanding of the risk 
# of each employee leaving, allow for threshold calibration, and might perform better on imbalanced data.
# '''

# # XGBoost Base Model Definition

# In[44]:


# Define the model
xgb_base = xgb.XGBRegressor(objective ='reg:squarederror')


# Note:
# The objective parameter defines the loss function that XGBoost will minimize
# 

# # XGBoost Hyperparameter Tuning

# I am establishing a function to determine the optimal set of hyperparameters 
# that yield the lowest negative mean squared error for the model. This approach 
# ensures a reusable framework for hyperparameter tuning of subsequent models:

# In[45]:


def tune_regressor_hyperparameters(reg, param_grid, X_train, y_train, scoring='neg_mean_squared_error', n_splits=3):
    '''
    This function optimizes the hyperparameters for a regressor by searching over a specified hyperparameter grid. 
    It uses GridSearchCV and cross-validation (KFold) to evaluate different combinations of hyperparameters. 
    The combination with the highest negative mean squared error is selected as the default scoring metric. 
    The function returns the regressor with the optimal hyperparameters.
    '''
    
    # Create the cross-validation object using KFold
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Create the GridSearchCV object
    reg_grid = GridSearchCV(reg, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    # Fit the GridSearchCV object to the training data
    reg_grid.fit(X_train, y_train)

    # Get the best hyperparameters
    best_hyperparameters = reg_grid.best_params_
    
    # Return best_estimator_ attribute which gives us the best model that has been fitted to the training data
    return reg_grid.best_estimator_, best_hyperparameters


# I'll set up the hyperparameters grid and utilize the tune_regressor_hyperparameters 
# function to pinpoint the optimal hyperparameters for our XGBoost regressor:

# In[46]:


# Define the parameters for grid search
xgb_param_grid = {
    'max_depth': [4, 5],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [200, 250, 300],
    'min_child_weight': [2, 3, 4]
}


# In[47]:


# Tune the hyperparameters
best_xgb, best_xgb_hyperparameters = tune_regressor_hyperparameters(xgb_base, xgb_param_grid, X_train, y_train)


# In[48]:


print('XGBoost Regressor Optimal Hyperparameters: \n', best_xgb_hyperparameters)


# # XGBoost Regressor Evaluation

# In[50]:


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):

    # Predict on training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics for training data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)

    # Calculate metrics for testing data
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    
    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame(data = [mae_test, mse_test, rmse_test, r2_test],
                              index = ['MAE', 'MSE', 'RMSE', 'R2 Score'],
                              columns = [model_name])
    # Print the metrics
    print(f"{model_name} Training Data Metrics:")
    print("MAE: {:.4f}".format(mae_train))
    print("MSE: {:.4f}".format(mse_train))
    print("RMSE: {:.4f}".format(rmse_train))
    print("R2 Score: {:.4f}".format(r2_train))
    
    print(f"\n{model_name} Testing Data Metrics:")
    print("MAE: {:.4f}".format(mae_test))
    print("MSE: {:.4f}".format(mse_test))
    print("RMSE: {:.4f}".format(rmse_test))
    print("R2 Score: {:.4f}".format(r2_test))
        
    return metrics_df


# In[51]:


xgb_result = evaluate_model(best_xgb, X_train, y_train, X_test, y_test, 'XGBoost')


# In[ ]:





# # CatBoost Base Model Definition

# In[52]:


# Define the model
ctb_base = CatBoostRegressor(verbose=0)


# # CatBoost Hyperparameter Tuning

# Afterward, I am setting up the hyperparameters grid and 
# utilize the tune_regressor_hyperparameters function to 
# pinpoint the optimal hyperparameters for our CatBoost regressor:

# In[53]:


# Define the parameters for grid search
ctb_param_grid = {
    'iterations': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
}


# In[54]:


# Tune the hyperparameters
best_ctb, best_ctb_hyperparameters = tune_regressor_hyperparameters(ctb_base, ctb_param_grid, X_train, y_train)


# In[55]:


print('\nCatBoost Regressor Optimal Hyperparameters: \n', best_ctb_hyperparameters)


# # CatBoost Regressor Evaluation

# Finally, I am evaluating the model's performance on both the training and test datasets using evaluate_model function:

# In[56]:


ctb_result = evaluate_model(best_ctb, X_train, y_train, X_test, y_test, 'CatBoost')


# The provided CatBoost model demonstrates a good performance on both the training and testing data. The difference in metrics (MAE, MSE, RMSE, R2 Score) between the two sets is minor, indicating a balanced model that is not significantly overfitting. The model shows a high R2 score of 0.9794 on the training data, indicating a good fit, and a slightly lower R2 score of 0.9359 on the testing data, suggesting decent generalization. The MAE, MSE, and RMSE on the test set are 0.0392, 0.0116, and 0.1078, respectively, which indicates that the model is making small errors on average. Overall, the model's performance is satisfactory and there is no strong evidence of overfitting. Additionally, the CatBoost model's performance is almost similar to the XGBoost model, making it a good alternative for this particular dataset.

# # Conclusion

# In[57]:


# Combine the dataframes
combined_df = pd.concat([ctb_result.T, xgb_result.T], axis=0)
combined_df['Model'] = ['CatBoost', 'XGBoost']

# Melt the dataframe
melted_df = combined_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

# Define custom colors
custom_colors = ['#009c05', 'darkorange']

# Create the barplot
plt.figure(figsize=(10,6))
sns.barplot(x='Score', y='Metric', hue='Model', data=melted_df, palette=custom_colors)

plt.title('Model Comparison')
plt.show()


# In[ ]:




