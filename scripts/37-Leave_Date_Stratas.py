#!/usr/bin/env python
# coding: utf-8

# ## Join Date Stratas

# In[1]:


# Data Manipulation
import numpy as np
import pandas as pd
import swifter
import scipy
from collections import Counter

# Auxilary
from tqdm import tqdm
from glob import glob
import json
import string
import time
import datetime
import os
import sys

# Plotting
import matplotlib.pyplot as plt
#%matplotlib inline


# ## Specify Subreddit

# In[2]:


if sys.argv[1] == '-f': 
    SUBREDDIT = 'GreatApes'
else:
    SUBREDDIT = sys.argv[1]
print(f"Subreddit: {SUBREDDIT}")


# ## Get Data

# In[3]:


# Get all files with user hate speech per day
files = glob(f'../final_date_hate_speech/{SUBREDDIT}/*.json')


# In[4]:


def columns_to_int(col_name):
    try:
        col_name = int(col_name)
        return col_name
    except:
        return col_name


# In[5]:


# Dict to store user data
users_df = pd.DataFrame()

# Keep only data around join date
keys_to_keep = [str(x) for x in range(-30, 30)] 
keys_to_keep.append('reddit_leave_date')
keys_to_keep.append('subreddit_join_date')

for file in tqdm(files):
    with open(file, 'r') as f_in:
        user_name = file.replace('\\', '/').split('/')[-1].split('.')[0]
        user_dict = json.load(f_in)
        user_dict['user_name'] = user_name
        user_df = pd.DataFrame(user_dict, index=[0])
        users_df = users_df.append(user_df, ignore_index=True)
users_df.columns = [columns_to_int(col) for col in users_df.columns]


# In[11]:


users_df['reddit_leave_date'] = pd.to_datetime(users_df['reddit_leave_date'], unit='s')
users_df['subreddit_join_date'] = pd.to_datetime(users_df['subreddit_join_date'], unit='s')
users_df['join_delta'] = users_df.apply(lambda row: int((row['reddit_leave_date'] - row['subreddit_join_date']).days), axis='columns')


# In[14]:


users_df1 = users_df[['user_name', 'subreddit_join_date', 'reddit_leave_date', 'join_delta']]
users_df2 = users_df.drop(columns=['user_name', 'subreddit_join_date', 'reddit_leave_date', 'join_delta'])
users_df2.columns = [columns_to_int(col) for col in users_df2.columns]
users_df2 = users_df2.sort_index(axis='columns')
users_df = pd.concat([users_df1, users_df2], axis='columns')


# In[15]:


# Extracting the quantiles
quantiles = users_df['join_delta'].quantile(np.arange(0, 1.01, 0.01), interpolation='higher')
quantiles.index = [round(idx,2) for idx in quantiles.index]

# Plotting the quantils
fig, ax = plt.subplots(figsize=(10,5))
quantiles.plot(kind='line', color='steelblue', label='')

# Demark the quantiles in increments of 0.05 and 0.25
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', s=100, label='Quantiles with 0.05 intervals')
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='red', s=100, label='Quantiles with 0.25 intervals')

# Titles, labels, legend
plt.title(f'Time Between Joining r/{SUBREDDIT} and Leaving Reddit', size=25, pad=40)
plt.xlabel('Quantile', size=20, labelpad=10)
plt.ylabel('Days', size=20, labelpad=10)
plt.legend(loc='best', fontsize=15)

# Axes
plt.xlim([-0.125, 1.125])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=15)

# Annotate the 0th, 25th, 50th, 75th and 100th percentiles
for x, y in zip(quantiles.index[::25], quantiles.values[::25]):
    plt.annotate(text=f'({x} , {y:.0f})', xy=(x,y), xytext=(x-0.1, y+max(quantiles.values)*0.05), fontweight='bold', fontsize=15)

plt.show()


# In[24]:


# Calculate mean hate speech in the 30 days range
str_filter = [col for col in users_df.columns if type(col)==str]
num_filter = [col for col in users_df.columns if type(col)==int and col >= 0]
col_filter = str_filter + num_filter
filtered_df = users_df[col_filter]
mean_hate_speech = users_df[num_filter].mean(axis='columns').fillna(0)
mean_hate_speech.name = 'mean_hate_speech'
filtered_df = pd.concat([mean_hate_speech, filtered_df], axis='columns')


# In[26]:


filtered_df.head(3)


# In[27]:


filtered_df[filtered_df.join_delta <= 365]['mean_hate_speech'].mean() # break-even is 214 days


# In[28]:


filtered_df[filtered_df.join_delta > 365]['mean_hate_speech'].mean()


# In[ ]:


# Get near users and save list
print('Early quit:')
print(filtered_df[filtered_df.join_delta <= 365]['mean_hate_speech'].mean())
near_users = filtered_df[filtered_df.join_delta <= 365]['user_name']
near_users.to_csv(f'{SUBREDDIT}_early_quit_users.csv', index=False)


# In[ ]:


# Get far users and save list
print('Late quit:')
print(filtered_df[filtered_df.join_delta > 365]['mean_hate_speech'].mean())
far_users = filtered_df[filtered_df.join_delta > 365]['user_name']
far_users.to_csv(f'{SUBREDDIT}_late_quit_users.csv', index=False)


# # End
