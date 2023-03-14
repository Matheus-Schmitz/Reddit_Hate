#!/usr/bin/env python
# coding: utf-8

# # Merge Missing User Data

# In[1]:


# Data Manipulation
import numpy as np
import pandas as pd

# Auxilary
from tqdm import tqdm
from glob import glob
import json
import os
import sys


# In[2]:


SUBREDDIT = sys.argv[1]
print(f"Subreddit: {SUBREDDIT}")


# ## User Lists

# #### Users to Merge Comments

# In[4]:


# Get all files with Treatment users' data
Treatment_json_files = glob(f'/effectcrawl/ISI/reddit/data/{SUBREDDIT}_Users_Comments/*.json') + glob(f'/effectcrawl/ISI/reddit/data/{SUBREDDIT}_Users_Submissions/*.json')
Treatment_json_files = [file.replace('\\','/') for file in Treatment_json_files]

# Extract the users from the filenames
Treatment_users = [file.split('/')[-1].replace('_comments.json', '') for file in Treatment_json_files]
Treatment_users = [user.replace('_submissions.json', '') for user in Treatment_users]
Treatment_users = list(set(Treatment_users))


# ## Merge Missing Comments

# In[ ]:


# Get all json files with comments for a given subreddit
comments_jsons = glob(f'/effectcrawl/ISI/reddit/data/{SUBREDDIT}_Comments/*.json')

# Iterate over the jsons with monthly subreddit data
for json_file in tqdm(comments_jsons):

    # Load month's data and get all authors
    subreddit_comments_df = pd.read_json(json_file, lines=True)
    users_to_iterate = subreddit_comments_df.author.unique().tolist()
    users_to_iterate = [user for user in users_to_iterate if user in Treatment_users and user != '[deleted]']

    for user in users_to_iterate:
        
        # Get the IDs of all posts attributed to that user in the subreddit data
        subreddit_user_comment_ids = subreddit_comments_df[subreddit_comments_df.author == user]['id'].values.tolist()
        
        # Load user data
        try:
            user_comments_df = pd.read_json(f'/effectcrawl/ISI/reddit/data/{SUBREDDIT}_Users_Comments/{user}_comments.json', lines=True)
        # If we have no data at all for that user, create a blank dataframe
        except:
            user_comments_df = pd.DataFrame(columns=['author', 'id'])
            
        original_size = user_comments_df.shape[0]

        # For each ID attributed to the author in the subreddit data, check if it is also on the author data
        for comment_id in subreddit_user_comment_ids:
            
            # If it's not, add that post to the author's data
            if 'id' in user_comments_df.columns.values.tolist() and comment_id not in user_comments_df['id'].values.tolist():
                missing_entry = subreddit_comments_df[subreddit_comments_df.id == comment_id].reset_index().iloc[0]
                user_comments_df = user_comments_df.append(missing_entry, ignore_index=True)
                
        # Once all comments were checked for a user, re-save the user data
        if user_comments_df.shape[0] > original_size:
            user_comments_df.to_json(f'/effectcrawl/ISI/reddit/data/{SUBREDDIT}_Users_Comments/{user}_comments.json', orient='records', lines=True)


# ## Merge Missing Submissions

# In[ ]:


# Get all json files with submissions for a given subreddit
submissions_jsons = glob(f'/effectcrawl/ISI/reddit/data/{SUBREDDIT}_Submissions/*.json')

# Iterate over the jsons with monthly subreddit data
for json_file in tqdm(submissions_jsons):

    # Load month's data and get all authors
    subreddit_submissions_df = pd.read_json(json_file, lines=True)
    users_to_iterate = subreddit_submissions_df.author.unique().tolist()
    users_to_iterate = [user for user in users_to_iterate if user in Treatment_users and user != '[deleted]']

    for user in users_to_iterate:
        
        # Get the IDs of all posts attributed to that user in the subreddit data
        subreddit_user_submission_ids = subreddit_submissions_df[subreddit_submissions_df.author == user]['id'].values.tolist()
        
        # Load user data
        try:
            user_submissions_df = pd.read_json(f'/effectcrawl/ISI/reddit/data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
        # If we have no data at all for that user, create a blank dataframe
        except:
            user_submissions_df = pd.DataFrame(columns=['author', 'id'])
            
        original_size = user_submissions_df.shape[0]

        # For each ID attributed to the author in the subreddit data, check if it is also on the author data
        for submission_id in subreddit_user_submission_ids:
            
            # If it's not, add that post to the author's data
            if 'id' in user_submissions_df.columns.values.tolist() and submission_id not in user_submissions_df['id'].values.tolist():
                missing_entry = subreddit_submissions_df[subreddit_submissions_df.id == submission_id].reset_index().iloc[0]
                user_submissions_df = user_submissions_df.append(missing_entry, ignore_index=True)
                
        # Once all submissions were checked for a user, re-save the user data
        if user_submissions_df.shape[0] > original_size:
            user_submissions_df.to_json(f'/effectcrawl/ISI/reddit/data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', orient='records', lines=True)

