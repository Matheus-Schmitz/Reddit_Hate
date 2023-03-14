#!/usr/bin/env python
# coding: utf-8

# # Crawl the Number of Subscribers per Subreddit




# In[1]:


import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import Counter
import json
import random
import requests
import re
import time
import os


# In[6]:


subreddits = ['CoonTown', 'fatpeoplehate', 'GreatApes', 'Incels', 'Braincels', 'WhiteRights', 'milliondollarextreme', 'MGTOW', 'honkler', 'frenworld']
#random.shuffle(subreddits)




# Regex pattern to find number of subs in the subredditstats.com page
find_sub_count = re.compile(r'"subscriberCount":(\d+)')


# In[14]:


subreddits_to_crawl = set()

for subreddit in subreddits:
    with open(f'{subreddit}_posts_per_subreddit.json', 'r') as json_file:
        posts_per_subreddit = json.load(json_file)
        subreddits_to_crawl.update(list(posts_per_subreddit.keys()))


# In[114]:


# Dict to store subs per subreddits (load already crawled data)
with open('subs_per_subreddit.json', 'r') as json_file:
    subs_per_subreddit = json.load(json_file)

# Count sequential crawling errors to implement extra sleep if needed
sequential_errors = 0

# Iterate over subreddits
for counter, subreddit in enumerate(tqdm(subreddits_to_crawl), 1):
    
    # If the data for a subreddit has already been crawled successfully, then pass
    if subreddit in subs_per_subreddit.keys() and type(subs_per_subreddit[subreddit]) == int:
        continue
    
    # Request the entire statistics page from subredditstats
    request = requests.get(f'https://subredditstats.com/r/{subreddit}')
    
    # Find the number of subs in that subreddits
    try:
        sub_count = int(find_sub_count.findall(request.text)[0])
        subs_per_subreddit[subreddit] = sub_count
        sequential_errors = 0
        
    # Banned subreddits (and 404 errors) won't return the number of subs
    except:
        subs_per_subreddit[subreddit] = 'error'
        print(f"Error on {subreddit} | Status code: {request.status_code}")
        sequential_errors += 1
    
    # Always sleep +-5 seconds
    time.sleep(3 + random.uniform(0, 4))
    
    # Every 50 requests wait an extra minute
    if counter % 50 == 0:
        time.sleep(60 + random.uniform(-5, 5))
        # Also re-save the file
        with open('subs_per_subreddit.json', 'w') as f_out:
            json.dump(subs_per_subreddit, f_out)
        
    # If we get 5+ errors in a row, then sleep more
    if sequential_errors >= 5:
        print(f'Sleeping Extra on {subreddit}')
        time.sleep(300)
        
    # If we get 15+ errors in a row, then stop
    if sequential_errors >= 15:
        print(f'Breaking on {subreddit}')
        break
    
# Once finished write results to json
with open('subs_per_subreddit.json', 'w') as f_out:
    json.dump(subs_per_subreddit, f_out)

