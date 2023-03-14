# Reddit_Hate

Scripts Explanation
01 - Scrapes the desired list of hateful subreddits
02 - Scrapes all users who posted on users from `01`
03 - Some user data might be missing posts from the hateful subreddit, this adds hateful subreddit posts back to the user history
04 - Counts how many posts each treatment user made in each subreddit
05 - Crawl subscriber counts for all subreddits
06 - Calculates which subreddits have the highest percentage of users from one of the treatment (hateful) subreddits
07 - Scrapes the top 30 subreddits from `06`
08 - Scrapes all users who posted on users from `07`
09 - Crawls random reddit posts to be used as speech baseline for SAGE
10 - Uses SAGE to generate a ranking of words most characterizing of each subreddit
24 - Computes speech for hateful users inside and outside their subreddit
25 - Generates plots with data from `24`
26 - Generate matching features for each user on monthly bins
27 - Matches users using features from `26`
30 - Plots a hate speech timeseries for various slicings of the data
31 - Calculates the instataneous hate speech increase upon becoming active in the hateful subreddit, using output from `49`
48 - Generate a timeseries of hate-speech per day per user for all users from `27`, considering only users on their first hateful subreddit
49 - Generates plots using data from `28`, filtering to keep only
51 - Calculates the instataneous hate speech increase upon becoming active in the hateful subreddit, using output from `52`
52 - Generates plots similar to `49`, but aggregating by hate category
54 - Plots a hate speech timeseries for various slicings of the data, aggregated by hate speech category
