#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# In[16]:


df=pd.read_excel("machine learning python loan approval.xlsx")


# In[17]:


df


# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.columns


# In[8]:


df.isnull().sum()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[18]:


df['LoanAmount'].hist(bins=20)


# In[23]:


df['loanAmount_log']=np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins=20)


# In[19]:


df.isnull().sum()


# In[20]:


df['ApplicantIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['ApplicantIncome_log']=np.log(df['ApplicantIncome'])
df['ApplicantIncome_log'].hist(bins=20)


# In[21]:


df.isnull().sum()


# In[24]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log=df.loanAmount_log.fillna(df.loanAmount_log.mean())

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

df.isnull().sum()


# In[31]:


#[0] df['Gender'].mode() might return a series if there are multipple modes .
#df['Gender'].mode()[0] ensure it will take the first mode. 


# In[25]:


x=df.iloc[:,np.r_[1:5,9:11,13:15]].values
y= df.iloc[:,12].values

x


# In[33]:


x.shape


# In[18]:


y


# In[26]:


print('per of missing gender is %2f%%' %((df['Gender'].isnull().sum()/df.shape[0])*100))


# In[27]:


print('number of peoplpe who take loan as group by gender')
print(df["Gender"].value_counts())
sns.countplot(x="Gender", data=df,palette="Set1")


# In[28]:


print('number of peoplpe who take loan as group by Martial status:')
print(df["Married"].value_counts())
sns.countplot(x="Married", data=df,palette="Set1")


# In[29]:


print('number of peoplpe who take loan as group by dependents')
print(df["Dependents"].value_counts())
sns.countplot(x="Dependents", data=df,palette="Set1")


# In[30]:


print('number of peoplpe who take loan as group by Self_employed')
print(df["Self_Employed"].value_counts())
sns.countplot(x="Self_Employed", data=df,palette="Set1")


# In[31]:


print('number of peoplpe who take loan as group by Loan Amount')
print(df["LoanAmount"].value_counts())
sns.countplot(x="LoanAmount", data=df,palette="Set1")


# In[32]:


print('number of peoplpe who take loan as group by Credit_History')
print(df["Credit_History"].value_counts())
sns.countplot(x="Credit_History", data=df,palette="Set1")


# In[33]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import LabelEncoder
Labelencoder_x=LabelEncoder


# In[34]:


X_train


# In[35]:


for i in range (0,5):
    X_train[:,i]=Labelencoder_x.fit_transform(X_train[:,i])
    X_train[:,7]=Labelencoder_x.fit_transform(X_train[:,7])
    
X_train   


# In[37]:


from sklearn.preprocessing import LabelEncoder
import numpy as np

# Assuming X_train is a 2D array where you want to encode columns 0 to 4 and column 7
for i in range(0, 5):
    # Convert column values to strings
    X_train[:, i] = X_train[:, i].astype(str)
    # Instantiate LabelEncoder
    labelencoder_x = LabelEncoder()
    # Fit and transform the column
    X_train[:, i] = labelencoder_x.fit_transform(X_train[:, i])

# Convert column 7 values to strings
X_train[:, 7] = X_train[:, 7].astype(str)
# Instantiate LabelEncoder
labelencoder_x = LabelEncoder()
# Fit and transform column 7
X_train[:, 7] = labelencoder_x.fit_transform(X_train[:, 7])


# In[38]:


X_train


# In[32]:


Labelencoder_y=LabelEncoder()
y_train=Labelencoder_y.fit_transform(y_train)
y_train


# In[39]:


from sklearn.preprocessing import LabelEncoder

# Assuming X_test is a 2D array where you want to encode columns 0 to 4 and column 7
for i in range(0, 5):
    # Instantiate LabelEncoder for each column
    labelencoder_x = LabelEncoder()
    # Convert column values to strings and fit-transform the column
    X_test[:, i] = labelencoder_x.fit_transform(X_test[:, i].astype(str))

# Instantiate LabelEncoder for column 7
labelencoder_x = LabelEncoder()
# Convert column 7 values to strings and fit-transform column 7
X_test[:, 7] = labelencoder_x.fit_transform(X_test[:, 7].astype(str))


# In[40]:


X_test


# In[41]:


Labelencoder_y=LabelEncoder()
y_test=Labelencoder_y.fit_transform(y_test)
y_test


# In[44]:


from sklearn.preprocessing import LabelEncoder

# Assuming X_test is a 2D array where you want to encode columns 0 to 4 and column 7
for i in range(0, 5):
    # Instantiate LabelEncoder for each column
    labelencoder_x = LabelEncoder()
    # Convert column values to strings, fit-transform the column, and reshape it to (-1, 1)
    X_test[:, i] = labelencoder_x.fit_transform(X_test[:, i].astype(str))

# Instantiate LabelEncoder for column 7
labelencoder_x = LabelEncoder()
# Convert column 7 values to strings, fit-transform column 7, and then flatten the array
X_test[:, 7] = labelencoder_x.fit_transform(X_test[:, 7].astype(str)).ravel()


# In[45]:


from sklearn.ensemble import RandomForestClassifier
rf_clf=RandomForestClassifier()
rf_clf.fit(X_train,y_train)


# In[63]:


# Assuming 'N' is 0 and 'Y' is 1
label_mapping = {'N': 0, 'Y': 1}

# Convert y_pred to numeric values using the mapping
y_pred1 = np.vectorize(label_mapping.get)(y_pred)



# In[64]:


from sklearn import metrics
y_pred = rf_clf.predict(X_test)
print('acc of random forest clf is',metrics.accuracy_score(y_pred1,y_test))
y_test


# In[51]:


y_test


# In[65]:


y_pred1


# In[ ]:




