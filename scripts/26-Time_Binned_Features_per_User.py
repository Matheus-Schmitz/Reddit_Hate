#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


SUBREDDIT = sys.argv[1]
print(f"Subreddit: {SUBREDDIT}")
LABEL = sys.argv[2]
print(f"Label: {LABEL}")


# ## Load Subreddits to Match On

# In[3]:


# Load csv with with subreddits ranked by ratio of Treatment members
member_ratios = pd.read_csv(f'Subreddits_with_high_{SUBREDDIT}_member_ratio.csv', index_col='subreddit')

# Keep the top 100 Treatment subreddits by total post count to use as a filter during feature generation
top_Treatment_Subreddits = member_ratios.nlargest(50, columns=['member_ratio']).index.to_list()
top_Treatment_Subreddits.append('Treatment')
top_Treatment_Subreddits = list(set(top_Treatment_Subreddits))


# ## Get Treatment Users and Files

# In[4]:


# Get all files with Treatment users' data
Treatment_json_files = glob(f'../data/{SUBREDDIT}_Users_Comments/*.json') + glob(f'../data/{SUBREDDIT}_Users_Submissions/*.json')
Treatment_json_files = [file.replace('\\','/') for file in Treatment_json_files]

# Extract the users from the filenames
Treatment_users = [file.split('/')[-1].replace('_comments.json', '') for file in Treatment_json_files]
Treatment_users = [user.replace('_submissions.json', '') for user in Treatment_users]
Treatment_users = list(set(Treatment_users))
    
# If stratifying, calculate the ratio of posts within the target subreddit
if LABEL != 'ALL':
    
    # First store all data in a dict
    Subreddit_Membership_Stregth_dict = {}

    for user in Treatment_users:

        target_counts = 0
        other_counts = 0

        # Load user comments and submissions
        try:
            df_comments = pd.read_json(f'../data/{SUBREDDIT}_Users_Comments/{user}_comments.json', lines=True)
            target_counts += df_comments[df_comments.subreddit == SUBREDDIT].shape[0]
            other_counts += df_comments[df_comments.subreddit != SUBREDDIT].shape[0]
        except:
            pass
        try:
            df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
            target_counts += df_submissions[df_submissions.subreddit == SUBREDDIT].shape[0]
            other_counts += df_submissions[df_submissions.subreddit != SUBREDDIT].shape[0]
        except:
            pass

        ratio = target_counts/(other_counts+target_counts)
        Subreddit_Membership_Stregth_dict[user] = {'target_counts': target_counts,
                                'other_counts': other_counts,
                                'ratio': ratio}
    
    # Save the dict as json
    with open(f"{SUBREDDIT}_membership_strength.json", "w") as outfile:
        json.dump(Subreddit_Membership_Stregth_dict, outfile)
    
    # Once finished create a DataFrame
    Subreddit_Membership_Stregth_df = pd.DataFrame(Subreddit_Membership_Stregth_dict).T
    
    # Then get usernames according to the desired strata
    if LABEL == 'LOW':
        Treatment_users = Subreddit_Membership_Stregth_df[Subreddit_Membership_Stregth_df.ratio < 0.03].index.tolist()
        print(f'Processing {len(Treatment_users)} {LABEL} users')
        
    elif LABEL == 'MEDIUM':
        Treatment_users = Subreddit_Membership_Stregth_df[(Subreddit_Membership_Stregth_df.ratio >= 0.03) & (Subreddit_Membership_Stregth_df.ratio < 0.10)].index.tolist()
    
    elif LABEL == 'HIGH':
        Treatment_users = Subreddit_Membership_Stregth_df[Subreddit_Membership_Stregth_df.ratio >= 0.10].index.tolist()
    
    else:
        raise Exception("Strata must be of of: ALL, LOW, MEDIUM, HIGH")
        
# Save stratified users:
np.savetxt(f'{SUBREDDIT}_{LABEL}_users.csv', Treatment_users, delimiter=',', fmt='%s', encoding='utf-8')


# In[ ]:


# Find bots in the Treatment user list
bot_substrings = ['bot','auto','transcriber',r'\[deleted\]','changetip','gif','bitcoin','tweet','messenger','mention','tube','link']
possible_bots = [user for user in Treatment_users if any(substring in user.lower() for substring in bot_substrings)]

# Remove the bots from the list
Treatment_users = list(set(Treatment_users) - set(possible_bots))
print(f"Treatment Users: {len(Treatment_users)}")

# Limit sample size to 3k treatments
Treatment_users = np.random.choice(Treatment_users, size=min(15000, len(Treatment_users)), replace=False)


# ## Get Control Users and Files

# In[6]:


# Get all files with Control users' data
Control_json_files = glob('../data/Control_Users_Comments/*.json') + glob('../data/Control_Users_Submissions/*.json')
Control_json_files = [file.replace('\\','/') for file in Control_json_files]

# Extract the users from the filenames
Control_users = [file.split('/')[-1].replace('_comments.json', '') for file in Control_json_files]
Control_users = [user.replace('_submissions.json', '') for user in Control_users]
Control_users = list(set(Control_users))


# In[7]:


# Find bots in the Treatment user list
bot_substrings = ['bot','auto','transcriber',r'\[deleted\]','changetip','gif','bitcoin']
possible_bots = [user for user in Control_users if any(substring in user.lower() for substring in bot_substrings)]

# Remove the bots from the list
Control_users = list(set(Control_users) - set(possible_bots))
print(f"Control Users: {len(Control_users)}")

# Limit sample size to a 5:1 ratio
Control_users = np.random.choice(Control_users, size=min(5*len(Treatment_users), len(Control_users)), replace=False)


# ## Get Users With Already Generated Features

# In[ ]:


# Ensure directories with processed data exsits
if not os.path.exists(f'../features/{SUBREDDIT}/treatment/'):
    os.makedirs(f'../features/{SUBREDDIT}/treatment/')

if not os.path.exists(f'../features/{SUBREDDIT}/control/'):
    os.makedirs(f'../features/{SUBREDDIT}/control/')
        
# Get usernames of users who have already been processed
Treatment_processed_jsons = glob(f'../features/{SUBREDDIT}/treatment/*.json')
Treatment_processed_jsons = [file.replace('\\','/') for file in Treatment_processed_jsons]
Treatment_processed = [file.split('/')[-1].replace('.json', '') for file in Treatment_processed_jsons]

# Get usernames of users who have already been processed
Control_processed_jsons = glob(f'../features/{SUBREDDIT}/Control/*.json')
Control_processed_jsons = [file.replace('\\','/') for file in Control_processed_jsons]
Control_processed = [file.split('/')[-1].replace('.json', '') for file in Control_processed_jsons]


# ## Get The Time of First Treatment Post For Treatment Users

# In[8]:


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


# In[9]:


# Convert timestamp to datetime
def get_datetime(timestamp):
    try:
        return datetime.datetime.fromtimestamp(timestamp)
    except:
        return None


# In[10]:


# Try to load the dataframe with precomputed values. Else compute the values and save as csv for later (takes ~25min)
print("Loading Treatment join dates.")
start = time.time()
df = pd.DataFrame(Treatment_users, columns=['treatment'])
df['first_Treatment_post_timestamp'] = df.treatment.swifter.progress_bar(False).apply(get_first_Treatment_post_date)
df['first_Treatment_post_datetime'] = df['first_Treatment_post_timestamp'].swifter.progress_bar(False).apply(get_datetime)
df.to_csv(f"{SUBREDDIT}_{LABEL}_Mark_0_per_User.csv", index=False)
end = time.time()
print(f"Time Taken: {(end-start)/60:.0f} minutes")


# In[11]:


# Get mark 0 dates (first Treatment post date)
mark_0_treatment = dict(zip(df.treatment, df.first_Treatment_post_datetime))


# In[12]:


# Check the earliest and late join time to know the range of months needed for binning
ealiest_datetime = min(list(mark_0_treatment.values()))
latest_datetime = max(list(mark_0_treatment.values()))
print(f"First join date: {ealiest_datetime}")
print(f"Last join date: {latest_datetime}")

# Increase the feature generation for 1 year before and after to account for possible new extremes as more data is crawled
ealiest_datetime = ealiest_datetime - datetime.timedelta(days=365)
latest_datetime = latest_datetime + datetime.timedelta(days=365)


# ### Create Monthly Bins for Data Slicing

# In[16]:


def months_between(start_date, end_date):
    """
    Given two instances of ``datetime.date``, generate a list of dates on
    the 1st of every month between the two dates (inclusive).

    e.g. "5 Jan 2020" to "17 May 2020" would generate:

        1 Jan 2020, 1 Feb 2020, 1 Mar 2020, 1 Apr 2020, 1 May 2020

    """
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} is not before end date {end_date}")

    year = int(start_date.year)
    month = int(start_date.month)

    while (year, month) <= (end_date.year, end_date.month):
        yield datetime.datetime(year, month, 1)

        # Move to the next month.  If we're at the end of the year, wrap around
        # to the start of the next.
        #
        # Example: Nov 2017
        #       -> Dec 2017 (month += 1)
        #       -> Jan 2018 (end of year, month = 1, year += 1)
        #
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1


# In[17]:


binned_months = []
for date in months_between(ealiest_datetime, latest_datetime):
    binned_months.append(date)


# ## Treatment Features Binned Monthly

# In[ ]:


# Loop over slices list of members
for user in tqdm(Treatment_users):
    
    # DF to store user data over all months
    df_treatment = pd.DataFrame()
    
    # If features for the user have already been computed, skip
    if user in Treatment_processed:
        continue
    
    # Load user comments and submissions
    try:
        _df_comments = pd.read_json(f'../data/{SUBREDDIT}_Users_Comments/{user}_comments.json', lines=True)
        # Delete empty data
        if _df_comments.shape[0] == 0:
            os.remove(f'../data/{SUBREDDIT}_Users_Comments/{user}_comments.json')
            _df_comments = pd.DataFrame(columns=['created_utc'])
    except:
        _df_comments = pd.DataFrame(columns=['created_utc'])
    try:
        _df_submissions = pd.read_json(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json', lines=True)
        # Delete empty data
        if _df_submissions.shape[0] == 0:
            os.remove(f'../data/{SUBREDDIT}_Users_Submissions/{user}_submissions.json')
            _df_submissions = pd.DataFrame(columns=['created_utc'])
    except:
        _df_submissions = pd.DataFrame(columns=['created_utc'])
    
    # Iterate over monthly bins to calcualte features up to the given month
    for dt in binned_months:
        df_comments = _df_comments[_df_comments.created_utc < dt.timestamp()] 
        df_submissions = _df_submissions[_df_submissions.created_utc < dt.timestamp()] 

        # Get account features
        account_name = user
        try:
            comments_date = int(df_comments.author_created_utc.min())
        except:
            comments_date = None
        try:
            submissions_date = int(df_submissions.author_created_utc.min())
        except:
            submissions_date = None
        if comments_date is not None and submissions_date is not None:
            account_date = min(comments_date, submissions_date)
        elif comments_date is not None:
            account_date = comments_date
        elif submissions_date is not None:
            account_date = submissions_date
        else:
            account_date = None
            
        comment_karma = df_comments.score.sum() if df_comments.shape[0]>0 else 0
        submission_karma = df_submissions.score.sum() if df_submissions.shape[0]>0 else 0
        account_karma = comment_karma + submission_karma
        account_submissions = df_submissions.shape[0]
        account_comments = df_comments.shape[0]
        comment_counts = df_comments[df_comments.subreddit.isin(top_Treatment_Subreddits)].subreddit.value_counts().to_dict() if 'subreddit' in df_comments and df_comments[df_comments.subreddit.isin(top_Treatment_Subreddits)].shape[0]>0 else dict()
        submission_counts = df_submissions[df_submissions.subreddit.isin(top_Treatment_Subreddits)].subreddit.value_counts().to_dict() if 'subreddit' in df_submissions and df_submissions[df_submissions.subreddit.isin(top_Treatment_Subreddits)].shape[0]>0 else dict()
        post_counts = dict(Counter(comment_counts) + Counter(submission_counts))

        # Createa a single dict with all features
        tmp_dict = post_counts.copy()
        tmp_dict['author']= account_name
        tmp_dict['creation_date']= account_date
        tmp_dict['karma']= account_karma
        tmp_dict['total_submissions']= account_submissions
        tmp_dict['total_comments']= account_comments

        # Convert dict to DF so it can be appended to existing data
        tmp_df = pd.DataFrame(tmp_dict, index=[0])

        # Find all subreddits with 0 post count and create columns for those
        null_subreddits = [subreddit for subreddit in top_Treatment_Subreddits if subreddit not in tmp_df.columns]
        tmp_df[null_subreddits] = 0
        
        # Add a column representing the Z-axis of months
        tmp_df['datetime'] = dt
        tmp_df['timestamp'] = int(dt.timestamp())

        # Ensure columns are always in the same order
        tmp_df.sort_index(axis=1, inplace=True)
       
        # Add to the df with all users on all months
        df_treatment = pd.concat([df_treatment, tmp_df], axis=0, ignore_index=True)
    
    # Save user's computed features
    if df_treatment.shape[0] > 0:
        df_treatment.to_csv(f'../features/{SUBREDDIT}/treatment/{user}.csv', index=False)


# ## Control Features Binned Monthly

# In[ ]:


# Loop over slices list of members
for user in tqdm(Control_users):
    
    # DF to store user data over all months
    df_treatment = pd.DataFrame()
    
    # If features for the user have already been computed, skip
    if user in Control_processed:
        continue
    
    # Load user comments and submissions
    try:
        _df_comments = pd.read_json(f'../data/Control_Users_Comments/{user}_comments.json', lines=True)
        # Delete empty data
        if _df_comments.shape[0] == 0:
            os.remove(f'../data/Control_Users_Comments/{user}_comments.json')
            _df_comments = pd.DataFrame(columns=['created_utc'])
        # Ignore users who posted on treatment subreddit
        if _df_comments[_df_comments.subreddit == SUBREDDIT].shape[0] > 0:
            continue
    except:
        _df_comments = pd.DataFrame(columns=['created_utc'])
    try:
        _df_submissions = pd.read_json(f'../data/Control_Users_Submissions/{user}_submissions.json', lines=True) 
        # Delete empty data
        if _df_submissions.shape[0] == 0:
            os.remove(f'../data/Control_Users_Submissions/{user}_submissions.json')
            _df_submissions = pd.DataFrame(columns=['created_utc'])
        # Ignore users who posted on treatment subreddit
        if _df_submissions[_df_submissions.subreddit == SUBREDDIT].shape[0] > 0:
            continue
    except:
        _df_submissions = pd.DataFrame(columns=['created_utc'])

    # Iterate over monthly bins to calcualte features up to the given month
    for dt in binned_months:
        df_comments = _df_comments[_df_comments.created_utc < dt.timestamp()] 
        df_submissions = _df_submissions[_df_submissions.created_utc < dt.timestamp()] 

        # Get account features
        account_name = user
        try:
            comments_date = int(df_comments.author_created_utc.min())
        except:
            comments_date = None
        try:
            submissions_date = int(df_submissions.author_created_utc.min())
        except:
            submissions_date = None
        if comments_date is not None and submissions_date is not None:
            account_date = min(comments_date, submissions_date)
        elif comments_date is not None:
            account_date = comments_date
        elif submissions_date is not None:
            account_date = submissions_date
        else:
            account_date = None
            
        comment_karma = df_comments.score.sum() if df_comments.shape[0]>0 else 0
        submission_karma = df_submissions.score.sum() if df_submissions.shape[0]>0 else 0
        account_karma = comment_karma + submission_karma
        account_submissions = df_submissions.shape[0]
        account_comments = df_comments.shape[0]
        comment_counts = df_comments[df_comments.subreddit.isin(top_Treatment_Subreddits)].subreddit.value_counts().to_dict() if 'subreddit' in df_comments and df_comments[df_comments.subreddit.isin(top_Treatment_Subreddits)].shape[0]>0 else dict()
        submission_counts = df_submissions[df_submissions.subreddit.isin(top_Treatment_Subreddits)].subreddit.value_counts().to_dict() if 'subreddit' in df_submissions and df_submissions[df_submissions.subreddit.isin(top_Treatment_Subreddits)].shape[0]>0 else dict()
        post_counts = dict(Counter(comment_counts) + Counter(submission_counts))

        # Createa a single dict with all features
        tmp_dict = post_counts.copy()
        tmp_dict['author']= account_name
        tmp_dict['creation_date']= account_date
        tmp_dict['karma']= account_karma
        tmp_dict['total_submissions']= account_submissions
        tmp_dict['total_comments']= account_comments

        # Convert dict to DF so it can be appended to existing data
        tmp_df = pd.DataFrame(tmp_dict, index=[0])

        # Find all subreddits with 0 post count and create columns for those
        null_subreddits = [subreddit for subreddit in top_Treatment_Subreddits if subreddit not in tmp_df.columns]
        tmp_df[null_subreddits] = 0
        
        # Add a column representing the Z-axis of months
        tmp_df['datetime'] = dt
        tmp_df['timestamp'] = int(dt.timestamp())

        # Ensure columns are always in the same order
        tmp_df.sort_index(axis=1, inplace=True)
                          
        # Add to the df with all users on all months
        df_treatment = pd.concat([df_treatment, tmp_df], axis=0, ignore_index=True)
    
    # Save user's computed features
    if df_treatment.shape[0] > 0:
        df_treatment.to_csv(f'../features/{SUBREDDIT}/control/{user}.csv', index=False)

