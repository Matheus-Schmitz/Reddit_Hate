#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import random
from pmaw import PushshiftAPI
api = PushshiftAPI()

MAX_AUTHORS_PER_SUBREDDIT = 10000


#subreddits = ['fatpeoplehate', 'CoonTown', 'GreatApes', 'Incels'] #, 'Braincel', 'NoNewNormal']
subreddits = [
'fatpeoplehate', 
'CoonTown', 
'GreatApes', 
'Incels',
#'NoNewNormal',
#
'Braincels',
'MGTOW',
'frenworld',
#'GenderCritical',
#'TruFemcels',
'honkler',
#'Neofag',
#'asktrp',
#'sjwhate',
#'trolling',
#'Delraymisfits',
#'CCJ2',
#'TrollGC',
#'TheRedPill',
#'FuckYou',
#'opieandanthony',
#'CringeAnarchy',
'WhiteRights',
#'ImGoingToHellForThis',
#'dolan',
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


for subreddit in subreddits:
    print(f"Subreddit: {subreddit}")
    authors = set()

    json_files = glob(f'/effectcrawl/ISI/reddit/data/{subreddit}_Comments/*.json') + glob(f'/effectcrawl/ISI/reddit/data/{subreddit}_Submissions/*.json')
 
    for json_file in json_files:
        df = pd.read_json(json_file, lines=True)
        file_authors = df.author.values.tolist()
        authors.update(file_authors)

    if not os.path.exists(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Comments/'):
        os.makedirs(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Comments/')
        
    if not os.path.exists(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Submissions/'):
        os.makedirs(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Submissions/')

    # Dont crawl bots
    bot_substrings = ['bot','auto','transcriber',r'\[deleted\]','changetip','gif','bitcoin','tweet','messenger','mention','tube','link']
    possible_bots = [user for user in authors if any(substring in user.lower() for substring in bot_substrings)]
    authors = set(authors) - set(possible_bots)

    # For each user who posted on GreatApes, get their entire post history
    for author in tqdm(authors):

        ### COMMENTS ###

        # Define the file name for the current request
        filename_comments = author + '_comments.json'

        # If the file has already been downloaded for that subreddit/day skip it
        downloaded_comments = os.listdir(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Comments/')
        if filename_comments not in downloaded_comments:

            try:
                # Fetch from Reddit
                comments = api.search_comments(author=author, limit=100000)

                # Convert to dataframe
                comments_df = pd.DataFrame(comments)   
                if comments_df.shape[0] > 0:   
                    comments_df.to_json(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Comments/' + filename_comments, orient='records', lines=True)   

            except:
                print(f"Error on {author} comments")


        ### SUBMISSIONS###

        # Define the file name for the current request
        filename_submissions = author + '_submissions.json'

        # If the file has already been downloaded for that subreddit/day skip it
        downloaded_submissions = os.listdir(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Submissions/')
        if filename_submissions not in downloaded_submissions:

            try:
                # Fetch from Reddit
                submissions = api.search_submissions(author=author, limit=100000)

                # Convert to dataframe
                submissions_df = pd.DataFrame(submissions)
                if submissions_df.shape[0] > 0:
                    submissions_df.to_json(f'/effectcrawl/ISI/reddit/data/{subreddit}_Users_Submissions/' + filename_submissions, orient='records', lines=True)
                    
            except:
                print(f"Error on {author} submissions")

        if len(downloaded_submissions) > MAX_AUTHORS_PER_SUBREDDIT and len(downloaded_comments) > MAX_AUTHORS_PER_SUBREDDIT:
            break
        elif len(downloaded_submissions) > 1.25 * MAX_AUTHORS_PER_SUBREDDIT or len(downloaded_comments) > 1.25 * MAX_AUTHORS_PER_SUBREDDIT:
            break