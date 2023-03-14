#!/usr/bin/env python
# coding: utf-8

# # Generate Vocabularies

# In[3]:


import sagee
from collections import Counter
import numpy as np
import pandas as pd
import string
from glob import glob
#import matplotlib.pyplot as plt
import sys
import time


# ## SAGE

# Github: https://github.com/jacobeisenstein/SAGE

# ### Auxilary Functions

# In[8]:


def word_counts_from_list_of_strings(lines):
    # Lowercase and remove newlines
    lines = [line.lower().replace('\n', ' ').strip() for line in lines if line is not None]

    # Remove subreddit tags
    lines = [line.replace(' r/', ' ').strip() for line in lines]
    lines = [line.replace(' /r/', ' ').strip() for line in lines]

    # Remove punctuation
    lines = [line.translate(str.maketrans(' ', ' ', string.punctuation)) for line in lines]

    # Remove URLs and tokenize
    words = [word for line in lines for word in line.split() if 'http' not in word]

    # Get word counts
    word_counts = Counter(words)
    
    return word_counts


# ### Baseline Counts

# In[3]:


print("Loading Baseline Counts")
start_time = time.time()

# Load baseline text
with open('random_reddit_data.csv', 'r', encoding="utf8") as f_in:
    lines = f_in.readlines()

# Get word counts
baseline_counts = word_counts_from_list_of_strings(lines)

# Time tracking
end_time = time.time()
print(f"Time Taken: {(end_time-start_time)/60:.0f} minutes" + "\n")


# ### Target Subreddit

# In[4]:


SUBREDDIT = sys.argv[1]
print(f"Subreddit: {SUBREDDIT}")


# In[ ]:


print("Loading Target Counts")
start_time = time.time()

# List monthly data files to load for this subreddit
comment_files = glob(f'../data/{SUBREDDIT}_Comments/*.json') 
submission_files = glob(f'../data/{SUBREDDIT}_Submissions/*.json')

lines = []

# Load comments
for json_file in comment_files:
    df = pd.read_json(json_file, lines=True)
    body = df[df.body != '[deleted]']['body'].values.tolist()
    lines.extend(body)

# Load submissions
for json_file in submission_files:
    df = pd.read_json(json_file, lines=True)
    title = df[df.title != '[deleted]']['title'].values.tolist()
    selftext = df[df.selftext != '[deleted]']['selftext'].values.tolist()
    lines.extend(title)
    lines.extend(selftext)
    
# Get word counts
target_counts = word_counts_from_list_of_strings(lines)

# Time tracking
end_time = time.time()
print(f"Time Taken: {(end_time-start_time)/60:.0f} minutes" + "\n")


# ### Generate Vocabulary

# In[ ]:


print("Running SAGE")
start_time = time.time()

# Generate vocabulary of top words
vocab = [word for word,count in target_counts.most_common(5000)]

# Get word counts for top words
x_baseline = np.array([baseline_counts[word] for word in vocab])
x_target = np.array([target_counts[word] for word in vocab])

# Compute the baseline log-probabilities of each word
mu = np.log(x_baseline) - np.log(x_baseline.sum())

# Run SAGE
eta = sagee.estimate(x_target,mu)

# Time tracking
end_time = time.time()
print(f"Time Taken: {(end_time-start_time)/60:.0f} minutes" + "\n")


# In[ ]:


# Get top words
top_words = sagee.topK(eta, vocab, K=300)
print(f"Original Vocabulary:")
print(len(top_words))
print(sorted(top_words))
print()


# In[ ]:


# Clean words
with open(f"{SUBREDDIT}_filter_out_words.txt", 'r') as f_in:
    words_to_remove = f_in.readlines()
    words_to_remove = [word.replace('\n', '') for word in words_to_remove]
    
for word in words_to_remove:
    if word in top_words:
        top_words.remove(word) 


# In[ ]:


print(f"High Precision Subset:")
print(len(top_words))
print(sorted(top_words))


# In[ ]:


# Save high precision lexicon
np.savetxt(SUBREDDIT+'_hate_words.csv', top_words, delimiter=',', fmt='%s', encoding='utf-8')


# ### Check Sparsity

# In[ ]:


# # Histogram to ensure we have sparisty
# fig, ax = plt.subplots(figsize=(15,4))
# plt.hist(eta,bins=200);
# plt.show()


# ## End
