#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# #### Top 30 Subreddits With Largest % of Treatment Members

# In[30]:

# tmp = pd.read_csv('random_subreddits.csv')
# subreddits = tmp['subreddit'].values.tolist()
# print(f'Crawling {len(subreddits)} subreddits')


# GreatApes
# subreddits = {'LiberalDegeneracy', 'ShadowBanned', 'promos', 'White_Pride',
#  'amishadowbanned', 'SRSsucks', 'abcqwerty123', 'sjsucks', 'RESissues',
#  'MensRants', 'Oppression', 'NorthAmerican', 'TrueRedditDrama',
#  'punchablefaces', 'TiADiscussion', 'RedditArmie', 'ukipparty', 'blackpower',
#  'gifrequests', 'lgv10', 'smokerslounge', 'test', 'GallowBoob', 'ebola',
#  'fullmovierequest', 'Stuff', 'ThePopcornStand', 'ads', 'utorrent',
#  'OurFlatWorld'}

# NoNewNormal
# subreddits = ['NoNewNormalBan', 'pan_media', 'CoronavirusCirclejerk', 'ChurchOfCOVID'
#  'EndTheLockdowns', 'FightingFakeNews', 'globeskepticism', 'ivermectin'
#  'AskConservatives', 'amishadowbanned', 'antimaskers', 'great_reset'
#  'TheBidenshitshow', 'TrueUnpopularOpinion', 'NEWPOLITIC', 'RealMichigan'
#  'walkaway', 'BlackMediaPresents', 'TimPool', 'CovidVaccine'
#  'CoincidenceTheorist', 'jimmydore', 'freeworldnews', 'VACCINES'
#  'FuckTheAltWrong', 'fragilecommunism', 'The_Chocker', 'TheLeftCantMeme'
#  'ModsAreKillingReddit', 'ShadowBanned']

# # Incels
# subreddits.update({'pan_media', 'promos', 'ForeverUnwanted', 'amishadowbanned', 'FA30plus',
#  'BPTmeta', 'ShadowBanned', 'redditsweats', 'test', 'OurFlatWorld',
#  'askredddit', 'NoNetNeutrality', 'RESissues', 'BannedFromThe_Donald',
#  'RoastMyHistory', 'TiADiscussion', 'the_meltdown', 'CircleOfTrustMeta',
#  'milliondollarextreme', 'HillaryForAmerica', 'MarchAgainstTrump', 'short',
#  'just_post', 'circlebroke2', 'gifrequests', 'assignedmale', 'wgtow',
#  'drunkenpeasants', 'EnoughTrumpSpam', 'CircleofTrust'})

# # fatpeoplehate
# subreddits.update({'promos', 'pan_media', 'RESissues', 'punchablefaces', 'amishadowbanned',
#  'uncensorship', 'ShadowBanned', 'RagenChastain', 'Stuff', 'thinnerbeauty',
#  'TiADiscussion', 'Fatsoshop', 'galaxys5', 'castmeas', 'INeedFeminismBecause',
#  'ads', 'test', 'sgsflair', 'SRSsucks', 'd3hardcore', 'yikyak',
#  'millionairemakers', 'redditsweats', 'TestPackPleaseIgnore', 'MensRants',
#  'gifrequests', 'fuckcoop', 'GallowBoob', 'abcqwerty123', 'wsgy'})

# # CoonTown
# subreddits.update({'promos', 'NorthAmerican', 'ShadowBanned', 'LiberalDegeneracy',
#  'amishadowbanned', 'Stuff', 'SRSsucks', 'uncensorship', 'ShitGhaziSays',
#  'ukipparty', 'RESissues', 'Oppression', 'sjsucks', 'punchablefaces', 'wsgy',
#  'GallowBoob', 'lolMorbidReality', 'TiADiscussion', 'lbregs', 'MensRants',
#  'pan_media', 'betternews', 'BernieSandersSucks', 'ads', 'test', 'allthingsmlg',
#  'joerogan2', 'RedditArmie', 'FabulousFerds', 'gifrequests'})

# subreddits = list(subreddits)
# random.shuffle(subreddits)


treatment_subs = ['CoonTown', 'fatpeoplehate', 'GreatApes', 'Incels', 'Braincels', 'WhiteRights', 'milliondollarextreme', 'MGTOW', 'honkler', 'frenworld']
random.shuffle(treatment_subs)

for subreddit in treatment_subs:
    subreddits = pd.read_csv(f'Subreddits_with_high_{subreddit}_member_ratio.csv')['subreddit']
    subreddits = list(subreddits)
    subreddits = subreddits[:30]
    random.shuffle(subreddits)



# In[ ]:


    # Generate timestamps for YearMonth dates
    dates = {}
    for year in range(2022, 2008, -1):
        for month in ['12', '11', '10', '09', '08', '07', '06', '05', '04', '03', '02', '01']:
            date = int(datetime.datetime(year,int(month),1,0,0).timestamp())
            dates[(year, month)] = date

    # Convert to list of tuples with YearMonth: Timestamp
    tuples = list(dates.items())


    # In[ ]:


    # Iterate through subreddits and crawl
    for subreddit in tqdm(subreddits):  
        print(f"Subreddit: {subreddit}")

        # Ensure directories for each subreddit exist
        directories = [f'/effectcrawl/ISI/reddit/data/{subreddit}_Comments', f'/effectcrawl/ISI/reddit/data/{subreddit}_Submissions']
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
                downloaded_files = os.listdir(f'../data/{subreddit}_Comments')
                if filename in downloaded_files:
                    continue

                # Fetch from Reddit
                comments = api.search_comments(subreddit=subreddit, limit=100000, 
                                                     after=tuples[idx+1][1], before=tuples[idx][1])

                # Save as csv
                sub_df = pd.DataFrame(comments)
                if sub_df.shape[0] > 0:
                    sub_df.to_json(f'/effectcrawl/ISI/reddit/data/{subreddit}_Comments/' + filename, orient='records', lines=True)

                #time.sleep(30)
                
            except:
                print(f"Error: Subreddit = {subreddit} | Tuple = {tuples[idx]}")

            ### SUBMISSIONS ###
        
            try:

                # Define the file name for the current request
                filename = subreddit + '_submissions_' + str(tuples[idx+1][0][0]) + '_' + str(tuples[idx+1][0][1]) + '.json'

                # If the file has already been downloaded for that subreddit/day skip it
                downloaded_files = os.listdir(f'../data/{subreddit}_Submissions')
                if filename in downloaded_files:
                    continue

                # Fetch from Reddit
                submissions = api.search_submissions(subreddit=subreddit, limit=100000, 
                                                     after=tuples[idx+1][1], before=tuples[idx][1])

                # Save as csv
                sub_df = pd.DataFrame(submissions)
                if sub_df.shape[0] > 0:
                    sub_df.to_json(f'/effectcrawl/ISI/reddit/data/{subreddit}_Submissions/' + filename, orient='records', lines=True)

                #time.sleep(30)
                
            except:
                print(f"Error: Subreddit = {subreddit} | Tuple = {tuples[idx]}")

