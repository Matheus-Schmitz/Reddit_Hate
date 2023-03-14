#!/usr/bin/env python
# coding: utf-8

# In[4]:


import datetime
import time
import sys
import requests
import json
import os
import copy
import pickle
import itertools
from tqdm import tqdm
import pandas as pd
import random
from pmaw import PushshiftAPI
api = PushshiftAPI()


subreddits = [
'fatpeoplehate', 
'CoonTown', 
'GreatApes', 
'Incels',
# 'NoNewNormal',
#
'Braincels',
'MGTOW',
'frenworld',
# 'GenderCritical',
# 'TruFemcels',
'honkler',
# 'Neofag',
# 'asktrp',
# 'sjwhate',
# 'trolling',
# 'Delraymisfits',
# 'CCJ2',
# 'TrollGC',
# 'TheRedPill',
# 'FuckYou',
# 'opieandanthony',
# 'CringeAnarchy',
'WhiteRights',
# 'ImGoingToHellForThis',
# 'dolan',
'milliondollarextreme',
#
# 'TopMindsOfReddit',
# 'DragonballLegends',
# 'PcBuild',
# 'lotrmemes',
# 'MakeNewFriendsHere',
# 'FASCAmazon',
# 'lostarkgame',
# 'Musicthemetime',
# 'SDSU',
# 'newsokunomoral',
# 'PergunteReddit',
# 'Kengan_Ashura',
# 'LegalAdviceUK',
# 'HungryArtists',
# 'awfuleverything',
# 'Pattaya',
# 'juggalo'
# 'SpiritualAwakening',
# 'AutoNewsPaper',
# 'DivinityOriginalSin',
# 'sfwtrees',
# 'MapPorn',
# 'IRS',
# 'PixelGun',
# 'starwarsmemes',
]

random.shuffle(subreddits)


storage_directory = '/effectcrawl/ISI/reddit/data/'


# Generate timestamps for YearMonth dates
dates = {}
for year in range(2022, 2008, -1):
    for month in ['12', '11', '10', '09', '08', '07', '06', '05', '04', '03', '02', '01']:
        date = int(datetime.datetime(year,int(month),1,0,0).timestamp())
        dates[(year, month)] = date

# Convert to list of tuples with YearMonth: Timestamp
tuples = list(dates.items())


# Iterate through subreddits and crawl
for subreddit in tqdm(subreddits):  
    print(f"Subreddit: {subreddit}")

    # Ensure directories for each subreddit exist
    directories = [f'{storage_directory}{subreddit}_Comments', f'{storage_directory}{subreddit}_Submissions']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Go through monthly TimeStamps and crawl
    for idx in range(len(tuples)-2):

        ### COMMENTS ###

        try:

            # Define the file name for the current request
            filename = subreddit + '_comments_' + str(tuples[idx+1][0][0]) + '_' + str(tuples[idx+1][0][1]) + '.json'

            # If the file has already been downloaded for that subreddit/day skip it
            downloaded_files = os.listdir(f'{storage_directory}{subreddit}_Comments')
            if filename in downloaded_files:
                continue

            # Fetch from Reddit
            comments = api.search_comments(subreddit=subreddit, limit=100000, 
                                                 after=tuples[idx+1][1], before=tuples[idx][1])

            # Save as csv
            sub_df = pd.DataFrame(comments)
            if sub_df.shape[0] > 0:
                sub_df.to_json(f'{storage_directory}{subreddit}_Comments/' + filename, orient='records', lines=True)

            #time.sleep(30)
            
        except:
            print(f"Error: Subreddit = {subreddit} | Tuple = {tuples[idx]}")

        ### SUBMISSIONS ###
    
        try:

            # Define the file name for the current request
            filename = subreddit + '_submissions_' + str(tuples[idx+1][0][0]) + '_' + str(tuples[idx+1][0][1]) + '.json'

            # If the file has already been downloaded for that subreddit/day skip it
            downloaded_files = os.listdir(f'{storage_directory}{subreddit}_Submissions')
            if filename in downloaded_files:
                continue

            # Fetch from Reddit
            submissions = api.search_submissions(subreddit=subreddit, limit=100000, 
                                                 after=tuples[idx+1][1], before=tuples[idx][1])

            # Save as csv
            sub_df = pd.DataFrame(submissions)
            if sub_df.shape[0] > 0:
                sub_df.to_json(f'{storage_directory}{subreddit}_Submissions/' + filename, orient='records', lines=True)

            #time.sleep(30)
            
        except:
            print(f"Error: Subreddit = {subreddit} | Tuple = {tuples[idx]}")
