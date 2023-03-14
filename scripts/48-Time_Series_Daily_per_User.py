#!/usr/bin/env python
# coding: utf-8

# # Time Series Daily

# In[1]:


import numpy as np
import pandas as pd
import swifter
import datetime
import os
import sys
from collections import Counter
import string
from functools import reduce
from tqdm import tqdm
import json
import time
from glob import glob


# ## Specify Subreddit

# In[2]:


SUBREDDIT = sys.argv[1]
print(f"Subreddit: {SUBREDDIT}")
LABEL = sys.argv[2]
print(f"Label: {LABEL}")


# ## Load Matched Pairs

# In[5]:


#df = pd.read_csv("Matched_Pairs.csv")
df = pd.read_csv(f"{SUBREDDIT}_{LABEL}_Manual_Matching.csv")
df.columns = ['treatment', 'control', 'distance']
df['treatment'] = df['treatment'].astype(str)
df['control'] = df['control'].astype(str)
df.shape


# In[6]:


# Check for bots in the matched data
df[df['treatment'].str.contains('|'.join(['bot','auto','transcriber',r'\[deleted\]','changetip','gif','bitcoin','tweet','messenger','mention','tube','link']))]


# In[7]:


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


# In[8]:


df = df[~df.treatment.isin(users_to_skip)]
print(df.shape)
df.head(3)


# ## List of Banned Subreddits

# In[9]:


banned_subreddits = [SUBREDDIT]
with open('banned_subreddits.txt', 'r') as f_in:
    lines = f_in.readlines()
    for line in lines:
        if 'unbanned' in line.lower(): # Skip unbanned
            continue
        if line[0] == '/': # convert /r/ to r/
            line = line[1:]
        if line[:2] == 'r/': # if its a subreddit add to the list
            subreddit = line.split()[0][2:] # first two characters are 'r/'
            banned_subreddits.append(subreddit)           
            
banned_subreddits = sorted(list(set(banned_subreddits)))
print(f"Found {len(banned_subreddits)} banned subreddits.")


# ## Get The Time of First Treatment Post For Treatment Users

# In[10]:


# Take a user id, parse through the crawled user history and return the timestamp for the first Treatment post
def get_first_Treatment_post_date(user):

    # Load user comments and submissions
    try:
        df_comments = pd.read_json(f'../data/{SUBREDDIT}_Users_Comments/{user}_comments.json', lines=True)
    except:
        df_comments = pd.DataFrame()
    try:
        df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
    except:
        df_submissions = pd.DataFrame()

    # Get first comment and first submission date, then use the ealiest one as the first post date
    first_Treatment_comment_date = df_comments[df_comments.subreddit == SUBREDDIT].created_utc.min() if 'subreddit' in df_comments and SUBREDDIT in set(df_comments.subreddit) else np.inf
    first_Treatment_submission_date = df_submissions[df_submissions.subreddit == SUBREDDIT].created_utc.min() if 'subreddit' in df_submissions and SUBREDDIT in set(df_submissions.subreddit) else np.inf
    first_Treatment_post_date = min(first_Treatment_comment_date, first_Treatment_submission_date)
    
    return first_Treatment_post_date


# In[11]:


# Convert timestamp to datetime
def get_datetime(timestamp):
    try:
        return datetime.datetime.fromtimestamp(timestamp)
    except:
        return None


# In[12]:


# Try to load the dataframe with precomputed values. Else compute the values and save as csv for later (takes ~25min)
print("Loading matched pair and getting the first avax post date.")
start = time.time()
# if os.path.exists('Matched_Pairs_With_Date.csv'):
#     df = pd.read_csv("Matched_Pairs_With_Date.csv")
#     df['first_Treatment_post_datetime'] = pd.to_datetime(df['first_Treatment_post_datetime'])
# else:
df['first_Treatment_post_timestamp'] = df.treatment.swifter.progress_bar(False).apply(get_first_Treatment_post_date)
df['first_Treatment_post_datetime'] = df['first_Treatment_post_timestamp'].swifter.progress_bar(False).apply(get_datetime)
df.to_csv(f"{SUBREDDIT}_{LABEL}_first_hateful_Matched_Pairs_With_Date.csv", index=False)
end = time.time()
print(f"Time Taken: {(end-start)/60:.0f} minutes")


# In[13]:


df.info()


# ## Load Hate Word Lexicon

# In[14]:


hate_word_lexicon = pd.read_csv(f'{SUBREDDIT}_hate_words.csv', header=None)
hate_word_lexicon = hate_word_lexicon[0].to_list()


# ### Split Treatment and Control

# In[15]:


# Also removing users with no Treatment data crawled
df_treatment = df[~(df.first_Treatment_post_timestamp == np.inf)].drop('control', axis=1)
df_control = df[~(df.first_Treatment_post_timestamp == np.inf)].drop('treatment', axis=1)


# ### Get Mark 0 Dates (The First Treatment post for each Treatment/Control Pair)

# In[16]:


mark_0_treatment = dict(zip(df.treatment, df.first_Treatment_post_datetime))
mark_0_control = dict(zip(df.control, df.first_Treatment_post_datetime))
mark_0_dates = {**mark_0_treatment, **mark_0_control}
mark_0_dates = {str(k):v for k, v in mark_0_dates.items()}


# ### Functions Needed for Generating Daily Counts

# In[17]:


def get_hate_word_count(Counter_obj):
    hate_word_dict = {k:v for k, v in dict(Counter_obj).items() if k in hate_word_lexicon}
    hate_word_count = sum(hate_word_dict.values())
    return hate_word_count


# In[18]:


def get_all_word_count(Counter_obj):
    all_word_count = sum(dict(Counter_obj).values())
    return all_word_count


# In[19]:


# For each post get the day in relation to mark_0, then get the hate word count in post and set a value in the form of:
# df[post, day_gap] = hate word count
def bin_hate_words_per_day(df_row):
    
    # Calculate the post gap in relation to the user's mark 0 (first Treatment post)
    row_datetime = get_datetime(df_row['created_utc'])
    user_mark_0 = mark_0_dates[str(df_row['author'])]
    timedelta_gap = row_datetime - user_mark_0
    gap_in_days = timedelta_gap.days // 1 # // 1 means aggregate per day, // 7 means aggregate per week
    
    # create a dict with {day_gap: count}
    index = df_row.name
    column = gap_in_days
    value = df_row.hate_word_count
    df_user_hate.at[index, column] = value


# In[20]:


# For each post get the day in relation to mark_0, then get the hate word count in post and set a value in the form of:
# df[post, day_gap] = all word count
def bin_all_words_per_day(df_row):
    
    # Calculate the post gap in relation to the user's mark 0 (first Treatment post)
    row_datetime = get_datetime(df_row['created_utc'])
    user_mark_0 = mark_0_dates[str(df_row['author'])]
    timedelta_gap = row_datetime - user_mark_0
    gap_in_days = timedelta_gap.days // 1  # // 1 means aggregate per day, // 7 means aggregate per week
    
    # create a dict with {day_gap: count}
    index = df_row.name
    column = gap_in_days
    value = df_row.all_word_count
    df_user_all.at[index, column] = value


# In[21]:


def reducer(accumulator, element):
    for key, value in element.items():
        accumulator[key] = accumulator.get(key, 0) + value
    return accumulator


# ### Generate Daily Hate Word Counts - Treatment Inside Treatment

# In[ ]:


# Ensure directory to store data exists
if not os.path.exists(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/'):
    os.makedirs(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/')


# In[27]:



# Filter users to keep only those who have the target subreddit as their first hateful subreddit
hateful_subreddits = [
'fatpeoplehate', 
'CoonTown', 
'GreatApes', 
'Incels',
'Braincels',
'MGTOW',
'frenworld',
'GenderCritical',
'TruFemcels',
'honkler',
'Neofag',
'asktrp',
'sjwhate',
'trolling',
'Delraymisfits',
'CCJ2',
'TrollGC',
'TheRedPill',
'FuckYou',
'opieandanthony',
'CringeAnarchy',
'WhiteRights',
'ImGoingToHellForThis',
'dolan',
'milliondollarextreme',
]

# Dont consider the target subreddit on the list of other hateful subreddits
hateful_subreddits.remove(SUBREDDIT)


users_to_ignore, users_to_consider, failed_users = set(), set(), set()

for user in tqdm(df_treatment.treatment.values):
    try:
        df_comments = pd.read_json(f'../data/{SUBREDDIT}_Users_Comments/{user}_comments.json', lines=True)
        if 'subreddit' not in df_comments.columns:
            df_comments['subreddit'] = None
    except:
        df_comments = pd.DataFrame(columns=['author', 'created_utc', 'subreddit'])
    try:
        df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
        if 'subreddit' not in df_submissions.columns:
            df_submissions['subreddit'] = None
    except:
        df_submissions = pd.DataFrame(columns=['author', 'created_utc', 'subreddit'])

    # Merge comments and submission into a single user dataframe
    df_user = pd.concat([df_submissions[['author', 'created_utc', 'subreddit']],
                         df_comments[['author', 'created_utc', 'subreddit']]],
                       ignore_index=True)
    df_user = df_user.sort_values('created_utc', ascending=True)

    # iterate though the ordered subreddits, and keep the user only if the target subreddit is the first hateful subreddit the used posted on
    for sub in df_user['subreddit'].unique():
        if sub == SUBREDDIT:
            users_to_consider.add(user)
            break
        elif sub in hateful_subreddits:
            users_to_ignore.add(user)
            break

print(f'# User on their first hateful subreddit: {len(users_to_consider)}')

# Use the list of users to consider to fetch the list of controls to consider
df_filtered = df[df['treatment'].isin(users_to_consider)]
controls_to_consider = df_filtered[~(df_filtered.first_Treatment_post_timestamp == np.inf)]['control'].tolist()
print('Generated filted set of users to consider!')




# Track users for which we cannot generate data (the df_user has size 0)
failed_users = list()

# Series to store all hate words for this subgroup
global_hate_word_series = pd.Series(dtype='float')

for user in tqdm(users_to_consider):
    
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
    
    # Handle users with no data (incomplete data crawling didn't fetch their Treatment posts)
    if df_user.shape[0] == 0:
        failed_users.append(user)
        continue
    
    # Author might have changed usernames over time and thus the variant names won't be on mark_0. 
    # Fix that by setting the author names in all posts
    author_last = str(df_user.author.iloc[-1])
    author_mode = str(df_user.author.mode()[0])
    if author_mode in mark_0_dates:
        df_user['author'] = author_mode
    elif author_last in mark_0_dates:
        df_user['author'] = author_last
    else:
        failed_users.append(user)
        continue
            
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
    _ = df_user_hate.apply(bin_hate_words_per_day, axis=1)
    _ = df_user_all.apply(bin_all_words_per_day, axis=1)
    
    # For each user, sum the total number of hatewords per day gap
    user_hate_words_per_day = df_user_hate[df_user_hate.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'hate_word_count'])].sum()
    user_all_words_per_day = df_user_all[df_user_all.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'all_word_count'])].sum()
    
    ############################################
    ### Calculate daily hate word percentage ###
    ############################################
    
    # Loop through days and calculate
    for day in user_hate_words_per_day.keys():
        hate_word_pct = user_hate_words_per_day[day] / user_all_words_per_day[day] if user_all_words_per_day[day] > 0 else 0
        user_hate_word_percentage_per_day[day] = hate_word_pct
        
    # Convert user's data to Series and append to the global series
    user_hate_words_series = pd.Series(user_hate_word_percentage_per_day, dtype='float')
    global_hate_word_series = pd.concat([global_hate_word_series, user_hate_words_series])
    
# After parsing all users, save the global_hate_word_series
global_hate_word_series.name = 'Hate_Word_Pct'
global_hate_word_series.to_csv(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_inside.csv', index=True)

# Report on users with errors
np.savetxt(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_inside_failed_users.csv', failed_users, delimiter=',', fmt="%s")
print(f"Failed on {len(failed_users)} users:")
print(failed_users)


# ### Generate Daily Hate Word Counts - Treatment Outside Treatment

# In[ ]:


# Ensure directory to store data exists
if not os.path.exists(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/'):
    os.makedirs(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/')


# In[ ]:


# Track users for which we cannot generate data (the df_user has size 0)
failed_users = list()

# Series to store all hate words for this subgroup
global_hate_word_series = pd.Series(dtype='float')

for user in tqdm(users_to_consider):
    
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
        df_comments = df_comments[df_comments.subreddit != SUBREDDIT] # Filter out Treatment posts
    except:
        df_comments = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext'])
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
    try:
        df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
        df_submissions = df_submissions[df_submissions.subreddit != SUBREDDIT] # Filter out Treatment posts
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
    
    # Handle users with no data (user might only have posted on Treatment and nowhere else)
    if df_user.shape[0] == 0:
        failed_users.append(user)
        continue
    
    # Author might have changed usernames over time and thus the variant names won't be on mark_0. 
    # Fix that by setting the author names in all posts
    author_last = str(df_user.author.iloc[-1])
    author_mode = str(df_user.author.mode()[0])
    if author_mode in mark_0_dates:
        df_user['author'] = author_mode
    elif author_last in mark_0_dates:
        df_user['author'] = author_last
    else:
        failed_users.append(user)
        continue
    
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
    _ = df_user_hate.apply(bin_hate_words_per_day, axis=1)
    _ = df_user_all.apply(bin_all_words_per_day, axis=1)
    
    # For each user, sum the total number of hatewords per day gap
    user_hate_words_per_day = df_user_hate[df_user_hate.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'hate_word_count'])].sum()
    user_all_words_per_day = df_user_all[df_user_all.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'all_word_count'])].sum()
    
    ############################################
    ### Calculate daily hate word percentage ###
    ############################################
    
    # Loop through days and calculate
    for day in user_hate_words_per_day.keys():
        hate_word_pct = user_hate_words_per_day[day] / user_all_words_per_day[day] if user_all_words_per_day[day] > 0 else 0 if user_all_words_per_day[day] > 0 else 0
        user_hate_word_percentage_per_day[day] = hate_word_pct
        
    # Convert user's data to Series and append to the global series
    user_hate_words_series = pd.Series(user_hate_word_percentage_per_day, dtype='float')
    global_hate_word_series = pd.concat([global_hate_word_series, user_hate_words_series])
    
# After parsing all users, save the global_hate_word_series
global_hate_word_series.name = 'Hate_Word_Pct'
global_hate_word_series.to_csv(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_outside.csv', index=True)

# Report on users with errors
np.savetxt(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_outside_failed_users.csv', failed_users, delimiter=',', fmt="%s")
print(f"Failed on {len(failed_users)} users:")
print(failed_users)


# ### Generate Daily Hate Word Counts - Treatment All Subreddits

# In[ ]:


# Ensure directory to store data exists
if not os.path.exists(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/'):
    os.makedirs(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/')


# In[ ]:


# Track users for which we cannot generate data (the df_user has size 0)
failed_users = list()

# Series to store all hate words for this subgroup
global_hate_word_series = pd.Series(dtype='float')

for user in tqdm(users_to_consider):
    
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
        df_comments = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext'])
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
    try:
        df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
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
    
    # Handle users with no data (user might only have posted on Treatment and nowhere else)
    if df_user.shape[0] == 0:
        failed_users.append(user)
        continue
    
    # Author might have changed usernames over time and thus the variant names won't be on mark_0. 
    # Fix that by setting the author names in all posts
    author_last = str(df_user.author.iloc[-1])
    author_mode = str(df_user.author.mode()[0])
    if author_mode in mark_0_dates:
        df_user['author'] = author_mode
    elif author_last in mark_0_dates:
        df_user['author'] = author_last
    else:
        failed_users.append(user)
        continue
    
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
    _ = df_user_hate.apply(bin_hate_words_per_day, axis=1)
    _ = df_user_all.apply(bin_all_words_per_day, axis=1)
    
    # For each user, sum the total number of hatewords per day gap
    user_hate_words_per_day = df_user_hate[df_user_hate.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'hate_word_count'])].sum()
    user_all_words_per_day = df_user_all[df_user_all.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'all_word_count'])].sum()
    
    ############################################
    ### Calculate daily hate word percentage ###
    ############################################
    
    # Loop through days and calculate
    for day in user_hate_words_per_day.keys():
        hate_word_pct = user_hate_words_per_day[day] / user_all_words_per_day[day] if user_all_words_per_day[day] > 0 else 0
        user_hate_word_percentage_per_day[day] = hate_word_pct
        
    # Convert user's data to Series and append to the global series
    user_hate_words_series = pd.Series(user_hate_word_percentage_per_day, dtype='float')
    global_hate_word_series = pd.concat([global_hate_word_series, user_hate_words_series])
    
# After parsing all users, save the global_hate_word_series
global_hate_word_series.name = 'Hate_Word_Pct'
global_hate_word_series.to_csv(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_all.csv', index=True)

# Report on users with errors
np.savetxt(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_all_failed_users.csv', failed_users, delimiter=',', fmt="%s")
print(f"Failed on {len(failed_users)} users:")
print(failed_users)


# ### Generate Daily Hate Word Counts - Treatment Non-Banned Subreddits

# In[ ]:


# Ensure directory to store data exists
if not os.path.exists(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/'):
    os.makedirs(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/')


# In[ ]:


# Track users for which we cannot generate data (the df_user has size 0)
failed_users = list()

# Series to store all hate words for this subgroup
global_hate_word_series = pd.Series(dtype='float')

for user in tqdm(users_to_consider):
    
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
        df_comments = df_comments[~df_comments.subreddit.isin(banned_subreddits)] # Filter out banned subreddits
    except:
        df_comments = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext'])
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
    try:
        df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
        df_submissions = df_submissions[~df_submissions.subreddit.isin(banned_subreddits)] # Filter out banned subreddits
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
    
    # Handle users with no data (user might only have posted on Treatment and nowhere else)
    if df_user.shape[0] == 0:
        failed_users.append(user)
        continue
    
    # Author might have changed usernames over time and thus the variant names won't be on mark_0. 
    # Fix that by setting the author names in all posts
    author_last = str(df_user.author.iloc[-1])
    author_mode = str(df_user.author.mode()[0])
    if author_mode in mark_0_dates:
        df_user['author'] = author_mode
    elif author_last in mark_0_dates:
        df_user['author'] = author_last
    else:
        failed_users.append(user)
        continue
    
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
    _ = df_user_hate.apply(bin_hate_words_per_day, axis=1)
    _ = df_user_all.apply(bin_all_words_per_day, axis=1)
    
    # For each user, sum the total number of hatewords per day gap
    user_hate_words_per_day = df_user_hate[df_user_hate.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'hate_word_count'])].sum()
    user_all_words_per_day = df_user_all[df_user_all.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'all_word_count'])].sum()
    
    ############################################
    ### Calculate daily hate word percentage ###
    ############################################
    
    # Loop through days and calculate
    for day in user_hate_words_per_day.keys():
        hate_word_pct = user_hate_words_per_day[day] / user_all_words_per_day[day] if user_all_words_per_day[day] > 0 else 0
        user_hate_word_percentage_per_day[day] = hate_word_pct
        
    # Convert user's data to Series and append to the global series
    user_hate_words_series = pd.Series(user_hate_word_percentage_per_day, dtype='float')
    global_hate_word_series = pd.concat([global_hate_word_series, user_hate_words_series])
    
# After parsing all users, save the global_hate_word_series
global_hate_word_series.name = 'Hate_Word_Pct'
global_hate_word_series.to_csv(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_notbanned.csv', index=True)

# Report on users with errors
np.savetxt(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_notbanned_failed_users.csv', failed_users, delimiter=',', fmt="%s")
print(f"Failed on {len(failed_users)} users:")
print(failed_users)


# ### Generate Daily Hate Word Counts - Treatment Banned Subreddits

# In[ ]:


# Ensure directory to store data exists
if not os.path.exists(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/'):
    os.makedirs(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/')


# In[ ]:


# Track users for which we cannot generate data (the df_user has size 0)
failed_users = list()

# Series to store all hate words for this subgroup
global_hate_word_series = pd.Series(dtype='float')

for user in tqdm(users_to_consider):
    
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
        df_comments = df_comments[df_comments.subreddit.isin(banned_subreddits)] # keep only banned subreddits
    except:
        df_comments = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext'])
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
    try:
        df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
        df_submissions = df_submissions[df_submissions.subreddit.isin(banned_subreddits)] # keep only banned subreddits
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
    
    # Handle users with no data (user might only have posted on Treatment and nowhere else)
    if df_user.shape[0] == 0:
        failed_users.append(user)
        continue
    
    # Author might have changed usernames over time and thus the variant names won't be on mark_0. 
    # Fix that by setting the author names in all posts
    author_last = str(df_user.author.iloc[-1])
    author_mode = str(df_user.author.mode()[0])
    if author_mode in mark_0_dates:
        df_user['author'] = author_mode
    elif author_last in mark_0_dates:
        df_user['author'] = author_last
    else:
        failed_users.append(user)
        continue
    
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
    _ = df_user_hate.apply(bin_hate_words_per_day, axis=1)
    _ = df_user_all.apply(bin_all_words_per_day, axis=1)
    
    # For each user, sum the total number of hatewords per day gap
    user_hate_words_per_day = df_user_hate[df_user_hate.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'hate_word_count'])].sum()
    user_all_words_per_day = df_user_all[df_user_all.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'all_word_count'])].sum()
    
    ############################################
    ### Calculate daily hate word percentage ###
    ############################################
    
    # Loop through days and calculate
    for day in user_hate_words_per_day.keys():
        hate_word_pct = user_hate_words_per_day[day] / user_all_words_per_day[day] if user_all_words_per_day[day] > 0 else 0
        user_hate_word_percentage_per_day[day] = hate_word_pct
        
    # Convert user's data to Series and append to the global series
    user_hate_words_series = pd.Series(user_hate_word_percentage_per_day, dtype='float')
    global_hate_word_series = pd.concat([global_hate_word_series, user_hate_words_series])
    
# After parsing all users, save the global_hate_word_series
global_hate_word_series.name = 'Hate_Word_Pct'
global_hate_word_series.to_csv(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_banned.csv', index=True)

# Report on users with errors
np.savetxt(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_banned_failed_users.csv', failed_users, delimiter=',', fmt="%s")
print(f"Failed on {len(failed_users)} users:")
print(failed_users)


# ### Generate Daily Hate Word Counts - Control

# In[ ]:


# Ensure directory to store data exists
if not os.path.exists(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/'):
    os.makedirs(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/')


# In[ ]:


# Track users for which we cannot generate data (the df_user has size 0)
failed_users = list()

# Series to store all hate words for this subgroup
global_hate_word_series = pd.Series(dtype='float')

for user in tqdm(controls_to_consider):
    
    # Dict to store hate word percentages for this user
    user_hate_word_percentage_per_day = {}
    
    #####################################
    ### Count Hate Words in Each post ###
    #####################################
    
    # Load user comments and submissions
    try:
        df_comments = pd.read_json(f'../data/Control_Users_Comments/{user}_comments.json', lines=True)
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
    except:
        df_comments = pd.DataFrame(columns=['author', 'created_utc', 'tokens', 'body', 'title', 'selftext'])
        df_comments['author'] = df_comments['author'].astype(str)
        df_comments['body'] = df_comments['body'].astype(str)
    try:
        df_submissions = pd.read_json(f'../data/Control_Users_Submissions/{user}_submissions.json', lines=True)
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
    
    # Handle users with no data (error during crawling)
    if df_user.shape[0] == 0:
        failed_users.append(user)
        continue
    
    # Author might have changed usernames over time and thus the variant names won't be on mark_0. 
    # Fix that by setting the author names in all posts
    author_last = str(df_user.author.iloc[-1])
    author_mode = str(df_user.author.mode()[0])
    if author_mode in mark_0_dates:
        df_user['author'] = author_mode
    elif author_last in mark_0_dates:
        df_user['author'] = author_last
    else:
        failed_users.append(user)
        continue
    
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
    _ = df_user_hate.apply(bin_hate_words_per_day, axis=1)
    _ = df_user_all.apply(bin_all_words_per_day, axis=1)
    
    # For each user, sum the total number of hatewords per day gap
    user_hate_words_per_day = df_user_hate[df_user_hate.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'hate_word_count'])].sum()
    user_all_words_per_day = df_user_all[df_user_all.columns.difference(['author', 'created_utc', 'tokens', 'word_counts', 'all_word_count'])].sum()
    
    ############################################
    ### Calculate daily hate word percentage ###
    ############################################
    
    # Loop through days and calculate
    for day in user_hate_words_per_day.keys():
        hate_word_pct = user_hate_words_per_day[day] / user_all_words_per_day[day] if user_all_words_per_day[day] > 0 else 0
        user_hate_word_percentage_per_day[day] = hate_word_pct
        
    # Convert user's data to Series and append to the global series
    user_hate_words_series = pd.Series(user_hate_word_percentage_per_day, dtype='float')
    global_hate_word_series = pd.concat([global_hate_word_series, user_hate_words_series])
    
# After parsing all users, save the global_hate_word_series
global_hate_word_series.name = 'Hate_Word_Pct'
global_hate_word_series.to_csv(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/control.csv', index=True)

# Report on users with errors
np.savetxt(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/control_failed_users.csv', failed_users, delimiter=',', fmt="%s")
print(f"Failed on {len(failed_users)} users:")
print(failed_users)


# ## End
