#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import sys
import random
from pmaw import PushshiftAPI
api = PushshiftAPI()


# In[2]:


# # GreatApes
# subreddits = {'LiberalDegeneracy', 'ShadowBanned', 'promos', 'White_Pride',
#  'amishadowbanned', 'SRSsucks', 'abcqwerty123', 'sjsucks', 'RESissues',
#  'MensRants', 'Oppression', 'NorthAmerican', 'TrueRedditDrama',
#  'punchablefaces', 'TiADiscussion', 'RedditArmie', 'ukipparty', 'blackpower',
#  'gifrequests', 'lgv10', 'smokerslounge', 'test', 'GallowBoob', 'ebola',
#  'fullmovierequest', 'Stuff', 'ThePopcornStand', 'ads', 'utorrent',
#  'OurFlatWorld'}

# # NoNewNormal
# # subreddits = ['NoNewNormalBan', 'pan_media', 'CoronavirusCirclejerk', 'ChurchOfCOVID'
# #  'EndTheLockdowns', 'FightingFakeNews', 'globeskepticism', 'ivermectin'
# #  'AskConservatives', 'amishadowbanned', 'antimaskers', 'great_reset'
# #  'TheBidenshitshow', 'TrueUnpopularOpinion', 'NEWPOLITIC', 'RealMichigan'
# #  'walkaway', 'BlackMediaPresents', 'TimPool', 'CovidVaccine'
# #  'CoincidenceTheorist', 'jimmydore', 'freeworldnews', 'VACCINES'
# #  'FuckTheAltWrong', 'fragilecommunism', 'The_Chocker', 'TheLeftCantMeme'
# #  'ModsAreKillingReddit', 'ShadowBanned']

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


# In[14]:


treatment_subs = ['CoonTown', 'fatpeoplehate', 'GreatApes', 'Incels', 'Braincels', 'WhiteRights', 'milliondollarextreme', 'MGTOW', 'honkler', 'frenworld']
random.shuffle(treatment_subs)

for ts in treatment_subs:
    subreddits = pd.read_csv(f'Subreddits_with_high_{ts}_member_ratio.csv')['subreddit']
    subreddits = list(subreddits)
    subreddits = subreddits[:30]
    random.shuffle(subreddits)



    authors = set()

    for subreddit in subreddits:

        print(subreddit)

        json_files = glob(f'/effectcrawl/ISI/reddit/data/{subreddit}_Comments/*.json') + glob(f'/effectcrawl/ISI/reddit/data/{subreddit}_Submissions/*.json')
     
        for json_file in json_files:
            df = pd.read_json(json_file, lines=True)
            df.dropna(subset=['author'], inplace=True)
            df['author'] = df['author'].astype(str)
            file_authors = df.author.values.tolist()
            authors.update(file_authors)


    # In[4]:


    if not os.path.exists('/effectcrawl/ISI/reddit/data/Control_Users_Comments/'):
        os.makedirs('/effectcrawl/ISI/reddit/data/Control_Users_Comments/')
        
    if not os.path.exists('/effectcrawl/ISI/reddit/data/Control_Users_Submissions/'):
        os.makedirs('/effectcrawl/ISI/reddit/data/Control_Users_Submissions/')


    # In[9]:


    # Dont crawl bots
    bot_substrings = ['bot','auto','transcriber',r'\[deleted\]','changetip','gif','bitcoin','tweet','messenger','mention','tube','link']
    possible_bots = [user for user in authors if any(substring in user.lower() for substring in bot_substrings)]
    authors = set(authors) - set(possible_bots)


    # In[ ]:


    # For each user who posted on GreatApes, get their entire post history
    for author in tqdm(authors):

        ### COMMENTS ###

        # Define the file name for the current request
        filename_comments = author + '_comments.json'

        # If the file has already been downloaded for that subreddit/day skip it
        downloaded_comments = os.listdir('/effectcrawl/ISI/reddit/data/Control_Users_Comments/')
        if filename_comments not in downloaded_comments:

            try:
                # Fetch from Reddit
                comments = api.search_comments(author=author, limit=100000)

                # Save as csv
                sub_df = pd.DataFrame(comments)
                if sub_df.shape[0] > 0:
                    sub_df.to_json('/effectcrawl/ISI/reddit/data/Control_Users_Comments/' + filename_comments, orient='records', lines=True)

            except:
                print(f"Error on {author} comments")


        ### SUBMISSIONS###

        # Define the file name for the current request
        filename_submissions = author + '_submissions.json'

        # If the file has already been downloaded for that subreddit/day skip it
        downloaded_submissions = os.listdir('/effectcrawl/ISI/reddit/data/Control_Users_Submissions/')
        if filename_submissions not in downloaded_submissions:

            try:
                # Fetch from Reddit
                submissions = api.search_submissions(author=author, limit=100000)

                # Save as csv
                sub_df = pd.DataFrame(submissions)
                if sub_df.shape[0] > 0:
                    sub_df.to_json('/effectcrawl/ISI/reddit/data/Control_Users_Submissions/' + filename_submissions, orient='records', lines=True)

            except:
                print(f"Error on {author} submissions")

