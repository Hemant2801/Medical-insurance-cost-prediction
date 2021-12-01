#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary dpendencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


# # Data collection and Analysis

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Medical insurance cost prediction/insurance.csv')


# In[3]:


#print the first 5 rows of the dataset
df.head()


# In[4]:


#print the last 5 rows of the dataset
df.tail()


# In[5]:


#to get some information about the dataset
df.info()


# In[6]:


#shape of the dataset
df.shape


# # Data analysis

# In[7]:


#statistical measure of the dataset
df.describe()


# In[8]:


#distribution of age
sns.set_style(style = 'darkgrid')
plt.figure(figsize = (8,8))
sns.displot(df['age'])
plt.title('AGE DISTRIBUTION')
plt.show()


# In[9]:


#For Gender column
plt.figure(figsize = (6,6))
sns.countplot(x = 'sex', data = df)
plt.title('GENDER DISTRIBUTION')
plt.show()


# In[10]:


df['sex'].value_counts()


# In[11]:


#BMI distribution
plt.figure(figsize = (8,8))
sns.displot(df['bmi'], kde = True)
plt.title('BMI DISTRIBUTION')
plt.show()


# In[12]:


#for children column
plt.figure(figsize = (6,6))
sns.countplot(x = 'children', data = df)
plt.title('CHILDREN')
plt.show()


# In[13]:


df['children'].value_counts()


# In[14]:


#for smoker
plt.figure(figsize = (6,6))
sns.countplot(x = 'smoker', data = df)
plt.title('SMOKING DISTRIBUTION')
plt.show()


# In[15]:


df['smoker'].value_counts()


# In[16]:


#region distribution
plt.figure(figsize = (6,6))
sns.countplot(x = 'region', data = df)
plt.title('REGION DISTRIBUTION')
plt.show()


# In[17]:


#Charges
plt.figure(figsize = (8,8))
sns.displot(df['charges'], kde = True)
plt.title('CHARGES')
plt.show()


# # Data preprocessing

# In[18]:


#encoding the categorical data
encoder = LabelEncoder()


# In[19]:


objlist = df.select_dtypes(include = 'object').columns
for col in objlist:
    df[col] = encoder.fit_transform(df[col].astype(str))


# In[20]:


df.head()


# Splitting the features and targets

# In[21]:


X = df.drop(columns = 'charges', axis = 1)
Y = df['charges']


# In[22]:


print(X.shape, Y.shape)


# # Splitting the data and Model evaluation

# In[23]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2, random_state = 2)


# In[24]:


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


# # Model training:
# 
# Linear Regression

# In[25]:


model = LinearRegression()


# In[26]:


model.fit(x_train, y_train)


# model evaluation :
# 
# R squared error

# In[27]:


#on training data
training_prediction = model.predict(x_train)

training_evaluation = metrics.r2_score(training_prediction, y_train)
print('R SQUARED ERROR FOR TRAINING DATA :', training_evaluation)


# In[28]:


#on testing data
testing_prediction = model.predict(x_test)

testing_evaluation = metrics.r2_score(testing_prediction, y_test)
print('R SQUARED ERROR FOR TESTING DATA :', testing_evaluation)


# XGB Regressor

# In[29]:


model1 = XGBRegressor()


# In[30]:


model1.fit(x_train, y_train)


# Model evaluation:
# 
# R Squared error

# In[31]:


#on training data
training_prediction = model1.predict(x_train)

training_evaluation = metrics.r2_score(training_prediction, y_train)
print('R SQUARED ERROR FOR TRAINING DATA :', training_evaluation)


# In[32]:


#on testing data
testing_prediction = model1.predict(x_test)

testing_evaluation = metrics.r2_score(testing_prediction, y_test)
print('R SQUARED ERROR FOR TESTING DATA :', testing_evaluation)


# # Building a predictive system

# sw = 3
# se = 2
# nw = 1
# ne = 0

# In[35]:


model_input = input()
input_list = [float(i) for i in model_input.split(',')]
input_array = np.asarray(input_list)
reshaped_array = input_array.reshape(1, -1)

ls = [model, model1]
for i in ls:
    prediction = i.predict(reshaped_array)
    print(f'THE PREDICTED VALUE FOR {i} :\n', prediction)


# In[ ]:




