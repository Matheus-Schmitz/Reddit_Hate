#!/usr/bin/env python
# coding: utf-8

# # Inside vs Outside Hate Word Distribution

# In[1]:


# Data Manipulation
import numpy as np
import pandas as pd
import swifter
import scipy
from collections import Counter

# Auxilary
from functools import reduce
from tqdm import tqdm
from glob import glob
import json
import string
import time
import datetime
import os
import sys


# ## Specify Subreddit

# In[2]:


SUBREDDIT = sys.argv[1]
print(f"Subreddit: {SUBREDDIT}")


# ## Load Lexicon

# In[3]:


hate_word_lexicon = pd.read_csv(f'{SUBREDDIT}_hate_words.csv', header=None)
hate_word_lexicon = hate_word_lexicon[0].to_list()


# ## Load Treatment Users

# In[4]:


# Get all files with Treatment users' data
Treatment_json_files = glob(f'../data/{SUBREDDIT}_Users_Comments/*.json') + glob(f'../data/{SUBREDDIT}_Users_Submissions/*.json')
Treatment_json_files = [file.replace('\\','/') for file in Treatment_json_files]

# Extract the users from the filenames
Treatment_users = [file.split('/')[-1].replace('_comments.json', '') for file in Treatment_json_files]
Treatment_users = [user.replace('_submissions.json', '') for user in Treatment_users]
Treatment_users = list(set(Treatment_users))

# Find bots in the Treatment user list
bot_substrings = ['bot','auto','transcriber',r'\[deleted\]','changetip','gif','bitcoin','tweet','messenger','mention','tube']
possible_bots = [user for user in Treatment_users if any(substring in user.lower() for substring in bot_substrings)]

# Remove the bots from the list
Treatment_users = list(set(Treatment_users) - set(possible_bots))
print(f"Treatment Users: {len(Treatment_users)}")

# Limit sample size to 3k treatments
Treatment_users = np.random.choice(Treatment_users, size=min(3000, len(Treatment_users)), replace=False)


# ## Functions for Generating Word Counts

# In[5]:


def get_hate_word_dict(Counter_obj):
    hate_word_dict = {k:v for k, v in dict(Counter_obj).items() if k in hate_word_lexicon}
    return hate_word_dict


# In[6]:


def reducer(accumulator, element):
    for key, value in element.items():
        accumulator[key] = accumulator.get(key, 0) + value
    return accumulator


# ## Inside Hate Words 

# In[7]:


hate_word_accumulator = dict()

for user in tqdm(Treatment_users):

    # Load user comments and submissions
    try:
        df_comments = pd.read_json(f'../data/{SUBREDDIT}_Users_Comments/{user}_comments.json', lines=True)
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
        df_comments = df_comments[df_comments.subreddit == SUBREDDIT] # Keep only Treatment posts
    except:
        df_comments = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext'])
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
    try:
        df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
        df_submissions = df_submissions[df_submissions.subreddit == SUBREDDIT] # Keep only Treatment posts
        df_submissions['author'] = df_submissions['author'].astype(str)
        df_submissions['selftext'] = df_submissions['selftext'].astype(str) # empty selftext was generating None instead of empty str
    except:
        df_submissions = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext'])
        df_submissions['author'] = df_submissions['author'].astype(str)
        df_submissions['selftext'] = df_submissions['selftext'].astype(str) # empty selftext was generating None instead of empty str

    # Remove punctionation and tokenize each posts' text
    df_comments['tokens'] = df_comments.body.swifter.progress_bar(False).apply(lambda text: str(text).lower().translate(str.maketrans('', '', string.punctuation)).split())
    df_submissions['title_tokens'] = df_submissions.title.swifter.progress_bar(False).apply(lambda text: str(text).lower().translate(str.maketrans('', '', string.punctuation)).split())
    df_submissions['selftext_tokens'] = df_submissions.selftext.swifter.progress_bar(False).apply(lambda text: str(text).lower().translate(str.maketrans('', '', string.punctuation)).split())
    df_submissions['tokens'] = df_submissions['title_tokens'] + df_submissions['selftext_tokens']


    # Merge comments and submission into a single user dataframe
    df_user = pd.concat([df_submissions[['author', 'created_utc', 'tokens']],
                         df_comments[['author', 'created_utc', 'tokens']]],
                       ignore_index=True)

    # Single list with all user texts
    user_texts = [word for lst in df_user['tokens'].values.tolist() for word in lst if word in hate_word_lexicon]

    # Parse through the word counts and get the number of hate words in each post
    hate_word_dict = Counter(user_texts)

    # Add the User's Count to the Global Count
    hate_word_accumulator = reduce(reducer, [hate_word_dict], hate_word_accumulator)
    
# Once finished write results to json
with open(f'{SUBREDDIT}_inside_hate_word_counts.json', 'w') as f_out:
    json.dump(hate_word_accumulator, f_out)


# ## Outside Hate Words

# In[8]:


hate_word_accumulator = dict()

for user in tqdm(Treatment_users):

    # Load user comments and submissions
    try:
        df_comments = pd.read_json(f'../data/{SUBREDDIT}_Users_Comments/{user}_comments.json', lines=True)
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
        df_comments = df_comments[df_comments.subreddit != SUBREDDIT] # Keep only non Treatment posts
    except:
        df_comments = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext'])
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
    try:
        df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
        df_submissions = df_submissions[df_submissions.subreddit != SUBREDDIT] # Keep only non Treatment posts
        df_submissions['author'] = df_submissions['author'].astype(str)
        df_submissions['selftext'] = df_submissions['selftext'].astype(str) # empty selftext was generating None instead of empty str
    except:
        df_submissions = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext'])
        df_submissions['author'] = df_submissions['author'].astype(str)
        df_submissions['selftext'] = df_submissions['selftext'].astype(str) # empty selftext was generating None instead of empty str

    # Remove punctionation and tokenize each posts' text
    df_comments['tokens'] = df_comments.body.swifter.progress_bar(False).apply(lambda text: str(text).lower().translate(str.maketrans('', '', string.punctuation)).split())
    df_submissions['title_tokens'] = df_submissions.title.swifter.progress_bar(False).apply(lambda text: str(text).lower().translate(str.maketrans('', '', string.punctuation)).split())
    df_submissions['selftext_tokens'] = df_submissions.selftext.swifter.progress_bar(False).apply(lambda text: str(text).lower().translate(str.maketrans('', '', string.punctuation)).split())
    df_submissions['tokens'] = df_submissions['title_tokens'] + df_submissions['selftext_tokens']


    # Merge comments and submission into a single user dataframe
    df_user = pd.concat([df_submissions[['author', 'created_utc', 'tokens']],
                         df_comments[['author', 'created_utc', 'tokens']]],
                       ignore_index=True)

    # Single list with all user texts
    user_texts = [word for lst in df_user['tokens'].values.tolist() for word in lst if word in hate_word_lexicon]

    # Parse through the word counts and get the number of hate words in each post
    hate_word_dict = Counter(user_texts)

    # Add the User's Count to the Global Count
    hate_word_accumulator = reduce(reducer, [hate_word_dict], hate_word_accumulator)
    
# Once finished write results to json
with open(f'{SUBREDDIT}_outside_hate_word_counts.json', 'w') as f_out:
    json.dump(hate_word_accumulator, f_out)


# # End
