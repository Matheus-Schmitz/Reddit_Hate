#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import Counter
import json
import random
import os


# ## Parse Through Treatment Users and Tally Their Posts per Subreddit

# In[2]:


# Skip obvious bots 
users_to_skip = [
    '[deleted]',
    'autowikibot',
    'TweetPoster',
    'imgurtranscriber',
    'SmallSubBot',
    'xkcd_transcriber',
    'image_linker_bot',
    '_youtubot_',
    'autoposting_system...1958',
    'autoposting_system',
    'autourbanbot',
    'guesses_gender_bot',
    'clay_davis_bot',
    'demobile_bot',
    '-faggotbot',
    'ObamaRobot',
    'mobile_link_fix_bot',
    'totes_meta_bot',
    'redditlinkfixerbot',
    'PayRespects-Bot',
    'RemindMeBot',
    'AutoModerator',
    'PORTMANTEAU-BOT',
    'Bot_Metric',
    'TotesMessenger',
    'auto-xkcd37',
    'imguralbumbot',
    'phonebatterylevelbot',
    'BooCMB',
    'CommonMisspellingBot',
    'BigLebowskiBot',
    'CakeDay--Bot',
    'ComeOnMisspellingBot',
    'icarebot',
    'HappyFriendlyBot',
    'dadjokes_bot',
]


bot_substrings = ['bot','auto','transcriber',r'\[deleted\]','changetip','gif','bitcoin','tweet','messenger','mention','tube','link']


# In[3]:


subreddits = ['CoonTown', 'fatpeoplehate', 'GreatApes', 'Incels', 'Braincels', 'WhiteRights', 'milliondollarextreme', 'MGTOW', 'honkler']
random.shuffle(subreddits)
subreddits = ['frenworld'] + subreddits


# In[4]:


for subreddit in subreddits:
    print(f'Subreddit: {subreddit}')
    subreddit_count = Counter(dict())

    json_files = glob(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Comments/*.json') + glob(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Submissions/*.json')

    # Go through each user and get a list of all subreddits it interacted with, then add up the counts
    for json_file in tqdm(json_files):
        try:
            df = pd.read_json(json_file, lines=True)
        except:
            print(f"Error on file: {json_file}")
            os.remove(json_file)
            continue
        if df.shape[0] == 0:
            os.remove(json_file)
            continue
        if df.author.mode()[0] in users_to_skip or any(substring in str(df.author.mode()[0]).lower() for substring in bot_substrings):
            os.remove(json_file)
            continue
        if 'subreddit' in df.columns:
            user_subreddits = Counter(df.subreddit.unique())
            subreddit_count.update(user_subreddits)


    # After getting the count, sort in descending order
    subreddit_count = {str(k): v for k, v in sorted(subreddit_count.items(), key=lambda item: item[1], reverse=True) if k is not None}

    # Save to json
    with open(f'{subreddit}_members_per_subreddit.json', 'w') as f_out:
        json.dump(subreddit_count, f_out)
    print()


# In[5]:


for subreddit in subreddits:
    print(f'Subreddit: {subreddit}')
    subreddit_count = Counter(dict())

    json_files = glob(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Comments/*.json') + glob(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Submissions/*.json')

    # Go through each user and count how many posts it made in each subreddit, then add up the counts
    for json_file in tqdm(json_files):
        try:
            df = pd.read_json(json_file, lines=True)
        except:
            print(f"Error on file: {json_file}")
            os.remove(json_file)
            continue
        if df.shape[0] == 0:
            os.remove(json_file)
            continue
        if df.author.mode()[0] in users_to_skip or any(substring in str(df.author.mode()[0]).lower() for substring in bot_substrings):
            os.remove(json_file)
            continue
        if 'subreddit' in df.columns:
            posts_per_subreddit = df.subreddit.value_counts().to_dict()
            posts_per_subreddit = Counter(posts_per_subreddit)
            subreddit_count.update(posts_per_subreddit)


    # After getting the count, sort in descending order
    subreddit_count = {str(k): v for k, v in sorted(subreddit_count.items(), key=lambda item: item[1], reverse=True) if k is not None}

    # Save to json
    with open(f'{subreddit}_posts_per_subreddit.json', 'w') as f_out:
        json.dump(subreddit_count, f_out)
    print()

