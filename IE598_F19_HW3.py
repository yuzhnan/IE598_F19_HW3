#!/usr/bin/env python
# coding: utf-8

# In[68]:


# Import data, initialization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
df = pd.read_csv('C:/Users/Thinkpad/Desktop/IE598/hw3/HY_Universe_corporate bond.csv', header=0)


# In[69]:


# Show the head, tail and summary of data
print(df.head())
print(df.tail())
summary=df.describe()
print(summary)


# In[70]:


# Genarate histogram of bond types
_ = plt.hist(df['bond_type'])
_ = plt.xlabel('bond type')
_ = plt.ylabel('number of stocks')
plt.show()


# In[71]:


# Generate a swarmplot that shows the distribution of coupon rates of three particular industries: oil, real estate, electric
df1=df.loc[(df['Industry']=='Real Estate')|(df['Industry']=='Oil Gas')|(df['Industry']=='Electric')]
_ = sns.swarmplot(x='Industry', y='Coupon', data=df1)
plt.show()


# In[72]:


# ECDF of Liquidity Score
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

x_liq,y_liq=ecdf(df['LiquidityScore'])
_=plt.plot(x_liq, y_liq, marker='.', linestyle='none')
_=plt.xlabel('Liquidity Score')
_=plt.ylabel('ECDF')
plt.show()


# In[73]:


# Generate a boxplot that shows the distribution of coupon rates of three particular industries: oil, real estate, electric
df1=df.loc[(df['Industry']=='Real Estate')|(df['Industry']=='Oil Gas')|(df['Industry']=='Electric')]
_ = sns.boxplot(x='Industry', y='Coupon', data=df1)
plt.show()


# In[74]:


_ = plt.plot(df['n_trades'], df['volume_trades'], marker='.', linestyle='none')
_ = plt.xlabel('total number of trades')
_ = plt.ylabel('total volume of trades')
plt.gcf().set_size_inches(15, 15)
plt.show()


# In[75]:


# Calculate the mean, median, variance and std of n_trades and volume_trades
print('mean value of number of trades:', np.mean(df['n_trades']),'\n')
print('median of number of trades:', np.median(df['n_trades']),'\n')
print('std of number of trades:', np.std(df['n_trades']),'\n')
print('variance of number of trades:', np.var(df['n_trades']),'\n')
print('mean value of volume of trades:', np.mean(df['volume_trades']),'\n')
print('median of volume of trades:', np.median(df['volume_trades']),'\n')
print('variance of volume of trades:', np.var(df['volume_trades']),'\n')
print('std of volume of trades:', np.std(df['volume_trades']),'\n')


# In[79]:


# Covariance of n_trades and volume_trades
# Compute the covariance matrix: covariance_matrix
covariance_matrix=np.cov(df['n_trades'],df['volume_trades'])
# Print covariance matrix
print(covariance_matrix)

# Extract covariance
trade_cov=covariance_matrix[0,1]

# Print the covariance
print('Covariance of n_trades and volume_trades:',trade_cov,'\n')


# In[80]:


# Pearson correlation coefficient of n_trades and volume_trades
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat=np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient
r=pearson_r(df['n_trades'],df['volume_trades'])

# Print the result
print('Pearson correlation coefficient of n_trades and volume_trades:',r,'\n')


# In[81]:


print("My name is {Yuzheng Nan}")
print("My NetID is: {ynan4}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

