#!/usr/bin/env python
# coding: utf-8

# # Average Hate Speech per Join Bandwidth

# In[13]:


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


# ## Specify Subreddit

# In[14]:


if sys.argv[1] == '-f': sys.argv[1] = 'GreatApes'
SUBREDDIT = sys.argv[1]
print(f"Subreddit: {SUBREDDIT}")


# ## Load Hate Lexicon

# In[15]:


hate_word_lexicon = pd.read_csv(f'{SUBREDDIT}_hate_words.csv', header=None)
hate_word_lexicon = hate_word_lexicon[0].to_list()


# ## Create Folder

# In[16]:


directory = f'../final_date_hate_speech/{SUBREDDIT}'


# In[17]:


if not os.path.exists(directory):
    os.makedirs(directory)


# ## Calculate hate speech and join date per user

# In[18]:


# Convert timestamp to datetime
def get_datetime(timestamp):
    try:
        return datetime.datetime.fromtimestamp(timestamp)
    except:
        print(f'Error on timestamp: {timestamp}')
        return None


# In[19]:


# Get all files with Treatment users' data
Treatment_json_files = glob(f'../data/{SUBREDDIT}_Users_Comments/*.json') + glob(f'../data/{SUBREDDIT}_Users_Submissions/*.json')
Treatment_json_files = [file.replace('\\','/') for file in Treatment_json_files]

# Extract the users from the filenames
Treatment_users = [file.split('/')[-1].replace('_comments.json', '') for file in Treatment_json_files]
Treatment_users = [user.replace('_submissions.json', '') for user in Treatment_users]
Treatment_users = list(set(Treatment_users))


# In[20]:


def get_hate_word_count(Counter_obj):
    hate_word_dict = {k:v for k, v in dict(Counter_obj).items() if k in hate_word_lexicon}
    hate_word_count = sum(hate_word_dict.values())
    return hate_word_count


# In[21]:


def get_all_word_count(Counter_obj):
    all_word_count = sum(dict(Counter_obj).values())
    return all_word_count


# In[22]:


# For each post get the day in relation to mark_0, then get the hate word count in post and set a value in the form of:
# df[post, day_gap] = hate word count
def bin_words_per_day(df_row, hate_only, subreddit_join_date):
    
    # Calculate the post gap in relation to the user's mark 0 (first Treatment post)
    row_datetime = get_datetime(df_row['created_utc'])
    user_mark_0 = get_datetime(subreddit_join_date)
    timedelta_gap = row_datetime - user_mark_0
    gap_in_days = timedelta_gap.days // 1 # // 1 means aggregate per day, // 7 means aggregate per week
    
    # create a dict with {day_gap: count}
    index = df_row.name
    column = gap_in_days
    if hate_only:
        value = df_row.hate_word_count
        df_user_hate.at[index, column] = value
    else:
        value = df_row.all_word_count
        df_user_all.at[index, column] = value


# In[ ]:


# Track users for which we cannot generate data (the df_user has size 0)
failed_users = list()

# Series to store all hate words for this subgroup
global_hate_word_series = pd.Series(dtype='float')

for user in tqdm(Treatment_users):
    
    # Dict to store hate word percentages for this user
    user_hate_word_percentage_per_day = {}
    
    #####################################
    ### Count Hate Words in Each post ###
    #####################################
    
    # Load user comments and submissions
    try:
        df_comments = pd.read_json(f'../data/{SUBREDDIT}_Users_Comments/{user}_comments.json', lines=True)
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
    except:
        df_comments = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext', 'subreddit'])
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
    try:
        df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
        df_submissions['author'] = df_submissions['author'].astype(str)
        df_submissions['selftext'] = df_submissions['selftext'].astype(str) # empty selftext was generating None instead of empty str
    except:
        df_submissions = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext', 'subreddit'])
        df_submissions['author'] = df_submissions['author'].astype(str)
        df_submissions['selftext'] = df_submissions['selftext'].astype(str) # empty selftext was generating None instead of empty str
    
    
    # Remove punctionation and tokenize each posts' text
    df_comments['tokens'] = df_comments.body.swifter.progress_bar(False).apply(lambda text: text.lower().translate(str.maketrans('', '', string.punctuation)).split())
    df_submissions['title_tokens'] = df_submissions.title.swifter.progress_bar(False).apply(lambda text: text.lower().translate(str.maketrans('', '', string.punctuation)).split())
    df_submissions['selftext_tokens'] = df_submissions.selftext.swifter.progress_bar(False).apply(lambda text: text.lower().translate(str.maketrans('', '', string.punctuation)).split())
    df_submissions['tokens'] = df_submissions['title_tokens'] + df_submissions['selftext_tokens']
    
    # Merge comments and submission into a single user dataframe
    df_user = pd.concat([df_submissions[['author', 'created_utc', 'tokens', 'subreddit']],
                         df_comments[['author', 'created_utc', 'tokens', 'subreddit']]],
                       ignore_index=True)
    
    
    # Handle users with no data (incomplete data crawling didn't fetch their Treatment posts)
    if df_user[df_user.subreddit == SUBREDDIT].shape[0] == 0 or df_user[df_user.subreddit != SUBREDDIT].shape[0] == 0:
        failed_users.append(user)
        continue
        
    # Get subreddit join date
    reddit_leave_date = df_user['created_utc'].max()
    subreddit_join_date = df_user[df_user.subreddit == SUBREDDIT]['created_utc'].min()
    join_delta = reddit_leave_date - subreddit_join_date
          
    # Then disregard inside posts
    df_user = df_user[df_user.subreddit != SUBREDDIT]
        
    # Get word count for all works in the tokenized texts (posts)
    df_user['word_counts'] = df_user['tokens'].swifter.progress_bar(False).apply(Counter)
    
    # Create one df for hatewords and one for all words to use MapReduce logic
    df_user_hate = df_user.copy(deep=True)
    df_user_all  = df_user.copy(deep=True)
    
    # Parse through the word counts and get the number of hate words in each post
    df_user_hate['hate_word_count'] = df_user_hate['word_counts'].swifter.progress_bar(False).apply(get_hate_word_count)
    df_user_all['all_word_count'] = df_user_all['word_counts'].swifter.progress_bar(False).apply(get_all_word_count)
    
    #####################################################
    ### Bin the Hate Word Counts Per day From Mark 0 ###
    #####################################################
    
    # Apply the function to write the dataframe in the form df[post_index, day_gap] = hate_word_count
    #_ = df_user_hate.swifter.progress_bar(False).apply(bin_hate_words_per_day, axis=1)
    _ = df_user_hate.apply(lambda row: bin_words_per_day(row, hate_only=True, subreddit_join_date=subreddit_join_date), axis='columns')
    _ = df_user_all.apply(lambda row: bin_words_per_day(row, hate_only=False, subreddit_join_date=subreddit_join_date), axis='columns')
    
    # For each user, sum the total number of hatewords per day gap
    user_hate_words_per_day = df_user_hate[df_user_hate.columns.difference(['author', 'created_utc', 'tokens', 'subreddit', 'word_counts', 'hate_word_count'])].sum()
    user_all_words_per_day = df_user_all[df_user_all.columns.difference(['author', 'created_utc', 'tokens', 'subreddit', 'word_counts', 'all_word_count'])].sum()
    
    ############################################
    ### Calculate daily hate word percentage ###
    ############################################
    
    # Loop through days and calculate
    for day in user_hate_words_per_day.keys():
        hate_word_pct = user_hate_words_per_day[day] / user_all_words_per_day[day] if user_all_words_per_day[day] > 0 else np.NaN
        user_hate_word_percentage_per_day[day] = hate_word_pct
        
    # Convert user's data to Series and append to the global series
    user_hate_word_percentage_per_day['reddit_leave_date'] = int(reddit_leave_date)
    user_hate_word_percentage_per_day['subreddit_join_date'] = int(subreddit_join_date)

    with open(directory+f'/{user}.json', 'w') as f_out:
        json.dump(user_hate_word_percentage_per_day, f_out)


# # End
