#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import numpy as np
import pandas as pd
from pmaw import PushshiftAPI
api = PushshiftAPI()


# In[2]:


import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# In[3]:


headers = {
    'User-Agent': 'USC-ISI',
    'From': 'mschmitz@isi.edu'
}


# In[ ]:


crawler_number = sys.argv[1]


# In[6]:


data = []
counter = 0

while True:
        
    # Sleep extra on failures
    try:
    
        # Get a random post
        submission = requests.request('GET', 'http://www.reddit.com/r/random.json?limit=1', headers=headers)

        # Flatt it so post and comments are accesible on the same 'path'
        flat = flatten(submission.json())

        # Then access all text (post and comments) and add to the data being crawled
        for child in flat['data_children']:
            text = child['data']['selftext']
            if text != '':
                text = text.replace('\n', '')
                text = text.replace(',', ';')
                text = text.strip()
                data.append(text)

        # Every 100 rounds save and sleep
        if counter >= 100:
            print(f'Saving {len(data)} posts', datetime.datetime.now())
            np.savetxt(f'random_reddit_data_{crawler_number}.csv', data, delimiter=",", fmt='%s')
            time.sleep(30)
            counter = 0
        
        counter += 1
        
    except:
        print('Error! Sleeping extra', datetime.datetime.now())
        time.sleep(100)
