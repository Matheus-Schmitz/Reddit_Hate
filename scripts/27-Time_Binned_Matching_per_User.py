#!/usr/bin/env python
# coding: utf-8

# # Time Binned Matching

# In[1]:


# Data Manipulation
import numpy as np
import pandas as pd
import json
import swifter
import scipy
from collections import Counter
from collections import defaultdict

# Auxilary
from tqdm import tqdm
from glob import glob
import string

# Dates
import time
import datetime
from dateutil.relativedelta import relativedelta

# System
import os
import sys

# Plot
import matplotlib.pyplot as plt
import seaborn as sns


# ## Specify Subreddit

# In[4]:


SUBREDDIT = sys.argv[1]
print(f"Subreddit: {SUBREDDIT}")
LABEL = sys.argv[2]
print(f"Label: {LABEL}")


# ## Load Treatment Mark 0

# In[6]:


# Load Treatment join dates
mark_0 = pd.read_csv(f'{SUBREDDIT}_{LABEL}_Mark_0_per_User.csv')

# Remove users without a join date (due to missing data when crawling)
mark_0.dropna(inplace=True)
mark_0['first_Treatment_post_timestamp'] = mark_0['first_Treatment_post_timestamp'].astype(int)
mark_0['first_Treatment_post_datetime'] = pd.to_datetime(mark_0['first_Treatment_post_datetime'])


# ## Load User's Features

# In[7]:


# Get usernames of control users who have already been processed
Control_processed = glob(f'../features/{SUBREDDIT}/control/*.csv')
Control_processed = [file.replace('\\','/') for file in Control_processed]
Control_processed = [file.split('/')[-1].replace('.csv', '') for file in Control_processed]
Control_processed = np.random.choice(Control_processed, size=min(15000, len(Control_processed)), replace=False)


# In[8]:


# Load all user's monthly features into a single DataFrame
df_treatment = pd.DataFrame(columns=['datetime'])
for user in tqdm(mark_0['treatment']):
    user_features = pd.read_csv(f'../features/{SUBREDDIT}/treatment/{user}.csv')
    df_treatment = pd.concat([df_treatment, user_features], ignore_index=True)
df_treatment['datetime'] = pd.to_datetime(df_treatment['datetime'])
df_treatment['author'] = df_treatment['author'].astype(str) # ensure users with numeric names have names as strings

df_control = pd.DataFrame(columns=['datetime'])
for user in tqdm(Control_processed):
    user_features = pd.read_csv(f'../features/{SUBREDDIT}/control/{user}.csv')
    df_control = pd.concat([df_control, user_features], ignore_index=True)
df_control['datetime'] = pd.to_datetime(df_control['datetime'])
df_control['author'] = df_control['author'].astype(str) # ensure users with numeric names have names as strings


# ## Mahalanobis Distance

# The most common estimands in non-experimental studies are the “average effect of the treatment on the treated” (ATT), which is the effect for those in the treatment group, and the “average treatment effect” (ATE), which is the effect on all individuals (treatment and control).

# If interest is in the ATT, Σ is the variance covariance matrix of X in the full control group; if interest is in the ATE then Σ is the variance covariance matrix of X in the pooled treatment and full control groups. If X contains categorical variables they should be converted to a series of binary indicators, although the distance works best with continuous variables.

# https://stackoverflow.com/questions/57475242/implementing-mahalanobis-distance-from-scratch-in-python  
# https://stackoverflow.com/questions/27686240/calculate-mahalanobis-distance-using-numpy-only

# In[9]:


# Natural logarithm with handling of non-positive values
def logscale(num):
    if num <= 0:
        return 0
    else:
        return np.log(num)


# In[10]:


# Specify features to ignore in matching
features_to_ignore = ['author', 'datetime', 'timestamp', 'creation_date']


# In[11]:


# Calculate the Inverse Correlation Matrix for each month bin
VI_dict = {}
features_to_remove = {}

for month in tqdm(df_treatment['datetime'].unique()):
    
    # Pool all users on that timespan
    control_pool = df_control[df_control['datetime']==month].drop(columns=features_to_ignore) 
    treatment_pool = df_treatment[df_treatment['datetime']==month].drop(columns=features_to_ignore) 
    pooled_users = pd.concat([treatment_pool, control_pool])
    
    # Log-scale features
    pooled_users = pooled_users.applymap(logscale)
    
    # Get all features with no variance (only 1 value across all samples)
    count_of_distinct_values_per_feature = defaultdict(lambda: [])
    for column in pooled_users.columns:
        size = len(pooled_users[column].value_counts())
        count_of_distinct_values_per_feature[size].append(column)
        
    # Store features to remove for that month
    features_to_remove[month] = count_of_distinct_values_per_feature[1].copy()
    
    # Get Features with perfect correlation
    mask = (pooled_users.corr() == 1).sum() > 1
    perf_corr = pooled_users.corr()[mask].index.values
    features_to_remove[month] += perf_corr.tolist()
        
    # Calculate the Inverse Covariance Matrix
    pooled_users = pooled_users.drop(columns=features_to_remove[month])
    V = np.cov(pooled_users, rowvar=False) #rowvar = False means variables are on columns
    VI = np.linalg.inv(V)
    
    # Store VI for that month
    VI_dict[month] = VI


# In[12]:


start_time = time.time()

# df to store matched
df_matched = pd.DataFrame(columns=['treatment', 'control', 'distance'])

# Build a queue of treament users to match
treatment_queue = list(mark_0['treatment'])

# Progress bar
pbar = tqdm(position=0, leave=True)

# Loop through users
while len(treatment_queue) > 0:
    
    # Sample a user
    user = np.random.choice(treatment_queue)
    treatment_queue.remove(user)  

    # Get the Treatment join date to know where to slice the dataframe
    timestamp = mark_0[mark_0.treatment == user]['first_Treatment_post_timestamp'].reset_index(drop=True)[0]
    dt = mark_0[mark_0.treatment == user]['first_Treatment_post_datetime'].reset_index(drop=True)[0]

    # Get the treatment features at the specific binned month
    treatment_user = df_treatment[(df_treatment['author']==user) & (df_treatment['timestamp']<timestamp)].tail(1)

    # If there is no user data pre-dating it joining the target subreddit, we cannot match that user
    if treatment_user['datetime'].shape[0] == 0:
        continue
    
    # Extract the latest month pre-joining to filter the candidate controls
    month = treatment_user['datetime'].values[0]

    # Shuffle df_control to avoid argmin match always picking the same lowest index user during ties
    df_control = df_control.sample(frac=1).reset_index(drop=True)

    # Get control features at the desired month
    control_candidates = df_control[df_control['datetime']==month]

    # Ensure the features considered match those on the inverse covariance matrix
    control_candidates = control_candidates.drop(columns=features_to_ignore+features_to_remove[month])
    treatment_user = treatment_user.drop(columns=features_to_ignore+features_to_remove[month])

    # Log-scale features
    control_candidates = control_candidates.applymap(logscale)
    treatment_user = treatment_user.applymap(logscale)
    
    # Fetch the Inverse Covariance Matrix
    VI = VI_dict[month]
    
    # Mahalanobis distance
    treatment_features = treatment_user.reset_index(drop=True).iloc[0].values
    distances = control_candidates.apply(lambda row: scipy.spatial.distance.mahalanobis(row, treatment_features, VI), axis=1)
    distances = distances.sample(frac=1)
    
    # Get the matched control
    match_idx = distances[distances == distances.min()].index[0]
    matched_control = df_control[df_control['datetime']==month].at[match_idx, 'author']
    df_matched = df_matched.append({'treatment': user,
                                    'control': matched_control,
                                    'distance': distances.min()}, ignore_index=True)
        
    # Update the progress bar
    pbar.update(1)
    
    # If the queue gets to size zero, repopulate it with treatments who got an identical control match 
    # While keeping the match for the treatment which achieved the lowest distance
    if len(treatment_queue) == 0:
        
        # Get the control users with more than one match
        controls = df_matched['control'].value_counts().index
        mask = df_matched['control'].value_counts() > 1
        duplicate_controls = controls[mask].to_list()
        print(f"Recomputing {len(duplicate_controls)} duplicate matches")
        
        # Iterate over them to find which treatment to keep and which to requeue
        for ctrl_user in duplicate_controls:
            
            # Get the distance of the closest match
            control_min_dist = df_matched[(df_matched['control']==ctrl_user)]['distance'].min()
            
            # Get the treatment_users to requeue
            control_matches = df_matched[(df_matched['control']==ctrl_user)]
            match_to_keep = control_matches[control_matches['distance']==control_min_dist]['treatment'].values.tolist()
            match_to_keep = np.random.choice(match_to_keep) # keep only 1 match if multiple have the same min dist
            requeue_treatments = df_matched[(df_matched['control']==ctrl_user) & (df_matched['treatment']!=match_to_keep)]['treatment'].values.tolist()
            
            # Remove the requeue_treatments from df_matched
            df_matched = df_matched[~df_matched['treatment'].isin(requeue_treatments)]
            
            # Remove the matched control from df_control
            df_control = df_control[df_control['author'] != ctrl_user]
                       
            # Readd the unmatched treatments to treatment_queue
            treatment_queue += requeue_treatments

# End tqdm tracking
pbar.close()

# Save matches
df_matched.to_csv(f"{SUBREDDIT}_{LABEL}_Manual_Matching.csv", index=False)

# Report time taken
end_time = time.time()
print(f"Finished in {(end_time-start_time)/60:.0f} minutes")


# # End
