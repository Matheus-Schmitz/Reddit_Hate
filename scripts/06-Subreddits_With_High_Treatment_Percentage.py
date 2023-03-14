#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import Counter
import json
import numpy as np
import random


# ## Specify a Subreddit

# In[2]:


subreddits = ['CoonTown', 'fatpeoplehate', 'GreatApes', 'Incels', 'Braincels', 'WhiteRights', 'milliondollarextreme', 'MGTOW', 'honkler', 'frenworld']



for subreddit in subreddits:
    print(f'Subreddit: {subreddit}')


    # ## Count Number of Members From Treatment Members in Each Subreddit

    # In[4]:


    # Dict to store counts per subreddits (load already crawled data)
    with open(f'{subreddit}_members_per_subreddit.json', 'r') as json_file:
        subreddit_member_count = json.load(json_file)
        
    print(f"{len(subreddit_member_count)}")
    print(subreddit_member_count)


    # ## Load Number of Subscribers to Each Subreddit

    # In[5]:


    # Dict to store subs per subreddits (load already crawled data)
    with open('subs_per_subreddit.json', 'r') as json_file:
        subs_per_subreddit = json.load(json_file)
        
    print(f"{len(subs_per_subreddit)}")
    print(subs_per_subreddit)


    # ## Load Data

    # #### Subs Per Subreddit

    # In[6]:


    # Conver to DataFrame
    df1 = pd.DataFrame.from_dict(subs_per_subreddit, orient='index', columns=['subs_per_subreddit'])

    # Remove rows with errors
    df1 = df1[df1.subs_per_subreddit.map(lambda x: isinstance(x,int))]

    # Calculate quartiles
    quantiles = df1.subs_per_subreddit.quantile(np.arange(0, 1.01, 0.25), interpolation='higher')
    quantiles.index = quantiles.index.values.round(2)
    print(quantiles)

    # Remove the bottom quartile as very low subscriber counts create noisy ratios
    df1 = df1[df1.subs_per_subreddit > quantiles[0.25]]
    df1.head(5)



    # #### Members Per Subreddit

    # In[8]:


    df3 = pd.DataFrame.from_dict(subreddit_member_count, orient='index', columns=['treatment_members_per_subreddit'])
    df3 = df3[df3.treatment_members_per_subreddit.map(lambda x: isinstance(x,int))]
    df3.head(5)


    # ## Calculate Ratios

    # #### Ratio with Members

    # In[12]:


    # Ratio with members
    df_members = df1.merge(df3, how='inner', left_index=True, right_index=True)
    df_members['member_ratio'] = df_members['treatment_members_per_subreddit']/df_members['subs_per_subreddit']
    df_members.sort_values('member_ratio', inplace=True, ascending=False)
    member_ratios = df_members.drop(columns=['subs_per_subreddit', 'treatment_members_per_subreddit'])
    member_ratios.reset_index(inplace=True, drop=False)
    member_ratios.columns=['subreddit', 'member_ratio']
    member_ratios.to_csv(f'Subreddits_with_high_{subreddit}_member_ratio.csv', index=False)


    # In[13]:


    df_members.head()


    # In[14]:


    print(df_members.head(30).index.values)

