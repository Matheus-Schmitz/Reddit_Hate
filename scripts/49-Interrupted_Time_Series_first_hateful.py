#!/usr/bin/env python
# coding: utf-8

# # Interrupted Time Series

# Guide 1: https://www.mdrc.org/sites/default/files/RDD%20Guide_Full%20rev%202016_0.pdf  
# Guide 2: https://bfi.uchicago.edu/wp-content/uploads/BFI_WP_201997.pdf  
# Guide 3: https://www.princeton.edu/~davidlee/wp/RDDEconomics.pdf  
# Guide 4: https://arxiv.org/pdf/1911.09511.pdf

# In[1]:


# Statistics
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.metrics import mean_squared_error

# Data Manipulation
import numpy as np
import pandas as pd

# File Manipulation
from glob import glob
from tqdm import tqdm
import json
import os
import sys

# Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#%matplotlib inline


# ## Specify Subreddit and Label

# In[2]:


if sys.argv[1] == '-f':
    SUBREDDIT = 'Braincels'
else:
    SUBREDDIT = sys.argv[1]
print(f'Subreddit: {SUBREDDIT}')
    
if '.json' in sys.argv[2]:
    LABEL = 'ALL'
else:
    LABEL = sys.argv[2]
print(f'LABEL: {LABEL}')
    
DAILY_AVERAGES = False
DOWNSAMPLE = False


# ## Load Data

# ### Treatment

# In[3]:


# Specify treatment subgroup
subgroup = 'outside'

# Load into dataframe
df_treatment = pd.read_csv(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/treatment_{subgroup}.csv', index_col=0)
if LABEL == 'near': df_treatment = df_treatment[df_treatment.index >= -31]
df_treatment = df_treatment[df_treatment['Hate_Word_Pct'].isna()==False]
df_treatment.sort_index(inplace=True)
if DAILY_AVERAGES:
    df_treatment = df_treatment.reset_index().groupby('index').mean()
print(f'Earliest sample: {df_treatment.index.min()}')
print(f'Latest sample: {df_treatment.index.max()}')
print(f'Unique days: {df_treatment.index.unique().shape[0]}')


# In[4]:


total_count = df_treatment.shape[0]
zero_count = df_treatment[df_treatment['Hate_Word_Pct'] == 0].shape[0]
nonzero_count = df_treatment[df_treatment['Hate_Word_Pct'] != 0].shape[0]
print(f"Total samples: {total_count} | {100 * total_count / total_count:.0f} %")
print(f"Zero samples: {zero_count} | {100 * zero_count / total_count:.0f} %")
print(f"Non-Zero samples: {nonzero_count} | {100 * nonzero_count / total_count:.0f} %")


# ### Control

# In[5]:


# Load into dataframe
df_control = pd.read_csv(f'../word_counts/{SUBREDDIT}/{LABEL}_first_hateful/control.csv', index_col=0)
df_control = df_control[df_control['Hate_Word_Pct'].isna()==False]
df_control.sort_index(inplace=True)
if DAILY_AVERAGES:
    df_control = df_control.reset_index().groupby('index').mean()
print(f'Earliest sample: {df_control.index.min()}')
print(f'Latest sample: {df_control.index.max()}')
print(f'Unique days: {df_control.index.unique().shape[0]}')


# In[6]:


total_count = df_control.shape[0]
zero_count = df_control[df_control['Hate_Word_Pct'] == 0].shape[0]
nonzero_count = df_control[df_control['Hate_Word_Pct'] != 0].shape[0]
print(f"Total samples: {total_count} | {100 * total_count / total_count:.0f} %")
print(f"Zero samples: {zero_count} | {100 * zero_count / total_count:.0f} %")
print(f"Non-Zero samples: {nonzero_count} | {100 * nonzero_count / total_count:.0f} %")


# ## Bandwidth Optimization

# In[7]:


# DF to store analysis results
coefficients = pd.DataFrame()
CI_05 = pd.DataFrame()
CI_95 = pd.DataFrame()
Pvalues = pd.DataFrame()
RMSEs = pd.DataFrame()

# Iterate over batches of 10 days
for bandwidth in tqdm(range(30, 365, 5)):
    
    # Lists to store values for RMSE calculateion
    truth = []
    preds = []
    
    # Loop over all test days to get RMSE
    for test_day in range(-1, max(-51, df_treatment.index.min()), -1):

        # Keep only data within the desired date range
        df_treatment_filtered = df_treatment[(df_treatment.index >= -bandwidth+test_day) & (df_treatment.index <= test_day)].copy()
        df_control_filtered = df_control[(df_control.index >= -bandwidth+test_day) & (df_control.index <= test_day)].copy()

        # Add class labels
        df_treatment_filtered['exposed'] = 1
        df_control_filtered['exposed'] = 0

        # Merge 
        df = pd.concat([df_treatment_filtered, df_control_filtered])

        # Then convert the index to a feature (time)
        df.reset_index(inplace=True)
        df.columns = ['time', 'percentage', 'exposed']
        df['time_x_exposed'] = df['time'] * df['exposed']

        # Split X and Y
        x = df[df['time'] < test_day].drop(columns='percentage')
        y = df[df['time'] < test_day]['percentage']

        # OLS Regression 
        model = sm.OLS(endog=y, exog=sm.add_constant(x)).fit()

        ### RMSE ###
        # Get the samples for testing (first day not included in model training)
        test_treatment = df[(df['time'] == test_day) & (df['exposed'] == 1)]
        test_control = df[(df['time'] == test_day) & (df['exposed'] == 0)]
        if test_treatment.shape[0] == 0 or test_control.shape[0] == 0:
            continue

        # Get X and Y for treatment and control at time -1
        x_test_treat = test_treatment.drop(columns='percentage')
        x_test_treat = [1] + x_test_treat.reset_index(drop=True).iloc[0].tolist() # [1] is the constant
        y_test_treat = test_treatment['percentage']
        y_test_treat = y_test_treat.values[0]

        x_test_contr = test_control.drop(columns='percentage')
        x_test_contr = [1] + x_test_contr.reset_index(drop=True).iloc[0].tolist() # [1] is the constant
        y_test_contr = test_control['percentage']
        y_test_contr = y_test_contr.values[0]

        # Predict
        pred_treat = model.predict(x_test_treat)[0]
        pred_contr = model.predict(x_test_contr)[0]
        
        # Add to lists for RMSE calculation
        truth.extend([y_test_treat, y_test_contr])
        preds.extend([pred_treat, pred_contr])
        
    # Calculate RMSE
    rmse = mean_squared_error(truth, preds, squared=False)
    results = pd.Series(rmse)
    results.name = bandwidth
    RMSEs = pd.concat([RMSEs, results], axis=1)
        
print(f"The optimal bandwidth according to cross-validation RMSE is {RMSEs.T[0].idxmin()} days.")


# In[8]:


# Create figure
fig, ax = plt.subplots(figsize=(10,5))

# Coefficients
ax.plot(RMSEs.T, label='RMSE', lw=3)
ax.scatter(x=RMSEs.T[0].idxmin(), y=RMSEs.T[0].min(), 
           color='C3', s=100, label=f'Optimal: {RMSEs.T[0].idxmin()} days')

# Text
plt.title(SUBREDDIT, size=30, pad=20)
plt.xlabel('Bandwidth', fontsize=30, weight='normal', labelpad=20)
plt.ylabel('', fontsize=30, weight='normal', labelpad=20)

# Axis
plt.legend(fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.locator_params(axis ='y', nbins=6)

# Show
plt.savefig(f"../figures_first_hateful/Bandwidth_Optimization_RMSE_{SUBREDDIT}_{LABEL}.png", bbox_inches='tight')
plt.show()


# ### Equalize Sample Frequencies

# In[9]:


if DOWNSAMPLE:

    # Get the optimal bandwidth
    optimal_bandwidth = RMSEs.T[0].idxmin()

    ### TREATMENT ###
    # Get data within the optimal bandwidth
    df_treatment_filtered = df_treatment[(df_treatment.index >= -optimal_bandwidth) & 
                                         (df_treatment.index < optimal_bandwidth)].copy()
    smallest_sample_size = df_treatment_filtered.reset_index().groupby('index').count().min().values[0]
    # Downsample all days to have the same amount of users
    df_treatment = df_treatment_filtered.reset_index().groupby('index').sample(smallest_sample_size, replace=False).set_index('index')

    ### CONTROL ###
    # Get data within the optimal bandwidth
    df_control_filtered = df_control[(df_control.index >= -optimal_bandwidth) & 
                                         (df_control.index < optimal_bandwidth)].copy()
    smallest_sample_size = df_control_filtered.reset_index().groupby('index').count().min().values[0]
    # Downsample all days to have the same amount of users
    df_control = df_control_filtered.reset_index().groupby('index').sample(smallest_sample_size, replace=False).set_index('index')


# ### Sensitivity Analysis

# In[10]:


# DF to store analysis results
coefficients = pd.DataFrame()
CI_05 = pd.DataFrame()
CI_95 = pd.DataFrame()
Pvalues = pd.DataFrame()

# Iterate over batches of 10 days
for bandwidth in tqdm(range(5, 365, 1)):
    
    # Keep only data within the desired date range
    df_treatment_filtered = df_treatment[(df_treatment.index >= -bandwidth) & (df_treatment.index < bandwidth)].copy()
    df_control_filtered = df_control[(df_control.index >= -bandwidth) & (df_control.index < bandwidth)].copy()

    # Add class labels
    df_treatment_filtered['exposed'] = 1
    df_control_filtered['exposed'] = 0
    
    # Split before and after
    df_treatment_before = df_treatment_filtered[df_treatment_filtered.index < 0].copy()
    df_treatment_after = df_treatment_filtered[df_treatment_filtered.index >= 0].copy()
    df_control_before = df_control_filtered[df_control_filtered.index < 0].copy()
    df_control_after = df_control_filtered[df_control_filtered.index >= 0].copy()
    df_treatment_before['interrupted'] = 0
    df_treatment_after['interrupted'] = 1
    df_control_before['interrupted'] = 0
    df_control_after['interrupted'] = 1
    
    # Merge 
    df = pd.concat([df_treatment_before, df_treatment_after, 
                    df_control_before, df_control_after])

    # Then convert the index to a feature (time)
    df.reset_index(inplace=True)
    df.columns = ['time', 'percentage', 'exposed', 'interrupted']
    df['time_x_exposed'] = df['time'] * df['exposed']
    df['time_x_interrupted'] = df['time'] * df['interrupted']
    df['exposed_x_interrupted'] = df['exposed'] * df['interrupted']
    df['time_x_exposed_x_interrupted'] = df['time'] * df['exposed'] * df['interrupted']
    
    # Split X and Y
    x = df.drop(columns='percentage')
    y = df['percentage']

    # OLS Regression 
    model = sm.OLS(endog=y, exog=sm.add_constant(x)).fit()
    
    # Get coefficients and confidence intervals
    model_coefs = model.params
    model_coefs.name = bandwidth
    coefficients = pd.concat([coefficients, model_coefs], axis=1)
    bottom_CI = model.conf_int()[0]
    bottom_CI.name = bandwidth
    CI_05 = pd.concat([CI_05, bottom_CI], axis=1)
    top_CI = model.conf_int()[1]
    top_CI.name = bandwidth
    CI_95 = pd.concat([CI_95, top_CI], axis=1)
    pvalues = model.pvalues
    pvalues.name = bandwidth
    Pvalues = pd.concat([Pvalues, pvalues], axis=1)


# In[11]:


# Create figure
fig, ax = plt.subplots(nrows=2, figsize=(20,10), sharex=True)

plt.suptitle(SUBREDDIT, size=45)

### Coefficients ###
plt.sca(ax[0])

# Lines and Error bars
ax[0].plot(coefficients.T, label=coefficients.T.columns, lw=3)
for idx, beta in enumerate(coefficients.T.columns):
    ax[0].fill_between(coefficients.T.index, 
                    CI_05.T[beta], 
                    CI_95.T[beta], 
                    facecolor=f'C{idx}', 
                    alpha=0.1,
                    interpolate=True)

# Axis
plt.xlabel('', fontsize=35, weight='normal', labelpad=20)
plt.ylabel('Coefficient', fontsize=35, weight='normal', labelpad=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].locator_params(axis ='y', nbins=6)


### p-values ###
plt.sca(ax[1])
ax[1].plot(Pvalues.T, lw=3)

# 0.05 Threshold
ax[1].axhline(y=0.05, color='red', xmin=0.025, xmax=0.975, alpha=0.75, label='95% Significance Threshold', ls='--')

# Axis
plt.xlabel('Bandwidth', fontsize=35, weight='normal', labelpad=20)
plt.ylabel('P-value', fontsize=35, weight='normal', labelpad=20)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].locator_params(axis ='y', nbins=6)


# Put a legend below the plot
plt.figlegend(loc='lower center', fontsize=35, ncol=3, bbox_to_anchor=(0.5, -0.3), frameon=False)

# Show
plt.savefig(f"../figures_first_hateful/Sensitivity_Analysis_Bandwidth_{SUBREDDIT}_{LABEL}.png", bbox_inches='tight')
plt.show()


# ### Samples per Day

# In[12]:


# Calculate the range of dates that will be used in falsification test
sampling_range = RMSEs.T[0].idxmin() * 1.5

# Select samples in the range from the optimal bandwidth
df_treatment_filtered = df_treatment[(df_treatment.index >= -sampling_range) & 
                                     (df_treatment.index < sampling_range)].copy()

df_control_filtered = df_control[(df_control.index >= -sampling_range) & 
                                 (df_control.index < sampling_range)].copy()

# Samples per day
treatment_samples_per_day = df_treatment_filtered.index.value_counts()
control_samples_per_day = df_control_filtered.index.value_counts()


# In[13]:


# Create figure
fig, ax = plt.subplots(figsize=(10,5))

# Add join date
plt.axvline(x=0, color='red', ymin=0.025, ymax=0.975, alpha=0.75, label=f'User Joined r/{SUBREDDIT}')

# Coefficients
ax.plot(treatment_samples_per_day.sort_index(), lw=3, color='C1', label=f'Outside r/{SUBREDDIT}')
ax.plot(control_samples_per_day.sort_index(), lw=3, color='C4', label='Control')

# Text
plt.title(SUBREDDIT, size=30, pad=20)
plt.xlabel('Day', fontsize=30, weight='normal', labelpad=20)
plt.ylabel('Num. Samples', fontsize=30, weight='normal', labelpad=20)

# Axis
plt.legend(fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.locator_params(axis ='y', nbins=8)

# Show
plt.savefig(f"../figures_first_hateful/Samples_per_Day_{SUBREDDIT}_{LABEL}.png", bbox_inches='tight')
plt.show()


# ## Falsification (Placebo) Tests

# Wald Test: https://stackoverflow.com/questions/50117157/how-can-i-do-test-wald-in-python

# In[14]:


# DF to store analysis results
coefficients = pd.DataFrame()
CI_05 = pd.DataFrame()
CI_95 = pd.DataFrame()
Pvalues = pd.DataFrame()
WALDs = pd.DataFrame()

# Fix the number of days and instead shift the treatment start point
bandwidth = RMSEs.T[0].idxmin()

# Test all possible days as breakpoint, but keep at least 20% of the data from the other period for evaluation stability
for shift in range(-round(bandwidth*0.5), +(round(bandwidth*0.5)+1), 1):  
    
    # Keep only data within the desired date range
    df_treatment_filtered = df_treatment[(df_treatment.index >= -bandwidth+shift) & (df_treatment.index < bandwidth+shift)].copy()
    df_control_filtered = df_control[(df_control.index >= -bandwidth+shift) & (df_control.index < bandwidth+shift)].copy()

    # Add class labels
    df_treatment_filtered['exposed'] = 1
    df_control_filtered['exposed'] = 0
    
    # Split before and after
    df_treatment_before = df_treatment_filtered[df_treatment_filtered.index < 0].copy()
    df_treatment_after = df_treatment_filtered[df_treatment_filtered.index >= 0].copy()
    df_control_before = df_control_filtered[df_control_filtered.index < 0].copy()
    df_control_after = df_control_filtered[df_control_filtered.index >= 0].copy()
    df_treatment_before['interrupted'] = 0
    df_treatment_after['interrupted'] = 1
    df_control_before['interrupted'] = 0
    df_control_after['interrupted'] = 1
    
    # Merge 
    df = pd.concat([df_treatment_before, df_treatment_after, 
                    df_control_before, df_control_after])

    # Then convert the index to a feature (time)
    df.reset_index(inplace=True)
    df.columns = ['time', 'percentage', 'exposed', 'interrupted']
    df['time_x_exposed'] = df['time'] * df['exposed']
    df['time_x_interrupted'] = df['time'] * df['interrupted']
    df['exposed_x_interrupted'] = df['exposed'] * df['interrupted']
    df['time_x_exposed_x_interrupted'] = df['time'] * df['exposed'] * df['interrupted']
    
    # Split X and Y
    x = df.drop(columns='percentage')
    y = df['percentage']

    # OLS Regression 
    model = sm.OLS(endog=y, exog=sm.add_constant(x)).fit()
    
    # Get coefficients and confidence intervals
    model_coefs = model.params
    model_coefs.name = shift
    coefficients = pd.concat([coefficients, model_coefs], axis=1)
    bottom_CI = model.conf_int()[0]
    bottom_CI.name = shift
    CI_05 = pd.concat([CI_05, bottom_CI], axis=1)
    top_CI = model.conf_int()[1]
    top_CI.name = shift
    CI_95 = pd.concat([CI_95, top_CI], axis=1)
    pvalues = model.pvalues
    pvalues.name = shift
    Pvalues = pd.concat([Pvalues, pvalues], axis=1)
    
    # Wald Test
    wald = model.wald_test(', '.join(x.columns.tolist()), scalar=True)
    result = pd.Series(wald.fvalue)
    result.name = shift
    WALDs = pd.concat([WALDs, result], axis=1)
    
    #print(f"Shift = {shift:>3} | R^2 = {model.rsquared:.3f} | MSE = {model.mse_model:.6f} | BIC = {model.bic:.0f} | Wald F-Statistic = {wald.fvalue:.0f} | Wald p-value = {wald.pvalue:.3f}")
    
print(f"The optimal breakpoint shift according to Wald Test's F-Statistic is {WALDs.T[0].idxmax()} days.")


# In[15]:


# Create figure
fig, ax = plt.subplots(figsize=(10,5))

# Coefficients
ax.plot(WALDs.T, label='Wald Test F-Statistic', lw=3)
ax.scatter(x=WALDs.T[0].idxmax(), y=WALDs.T[0].max(), 
           color='C3', s=100, label=f'Optimal: {WALDs.T[0].idxmax()} days')

# Add join date
plt.axvline(x=0, color='red', ymin=0.025, ymax=0.975, alpha=0.75, label=f'User Joined r/{SUBREDDIT}')

# Text
plt.title(SUBREDDIT, size=30, pad=20)
plt.xlabel('Breakpoint Shift', fontsize=30, weight='normal', labelpad=20)
plt.ylabel('', fontsize=30, weight='normal', labelpad=20)

# Axis
plt.legend(fontsize=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.locator_params(axis ='y', nbins=6)

# Show
plt.savefig(f"../figures_first_hateful/Falsification_Wald_FStatistic_{SUBREDDIT}_{LABEL}.png", bbox_inches='tight')
plt.show()


# ### Sensitivity Analysis

# In[16]:


# Create figure
fig, ax = plt.subplots(nrows=2, figsize=(20,10), sharex=True)

plt.suptitle(SUBREDDIT, size=40)

### Coefficients ###
plt.sca(ax[0])

# Lines and Error bars
ax[0].plot(coefficients.T, label=coefficients.T.columns, lw=3)
for idx, beta in enumerate(coefficients.T.columns):
    ax[0].fill_between(coefficients.T.index, 
                        CI_05.T[beta], 
                        CI_95.T[beta], 
                        facecolor=f'C{idx}', 
                        alpha=0.1,
                        interpolate=True)

# Axis
plt.xlabel('', fontsize=30, weight='normal', labelpad=20)
plt.ylabel('Coefficient', fontsize=30, weight='normal', labelpad=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].locator_params(axis ='y', nbins=6)


### p-values ###
plt.sca(ax[1])
ax[1].plot(Pvalues.T, lw=3)

# 0.05 Threshold
ax[1].axhline(y=0.05, color='red', xmin=0.025, xmax=0.975, alpha=0.75, label='95% Significance Threshold', ls='--')

# Axis
plt.xlabel('Breakpoint Shift', fontsize=30, weight='normal', labelpad=20)
plt.ylabel('P-value', fontsize=30, weight='normal', labelpad=20)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].locator_params(axis ='y', nbins=6)

# Put a legend below the plot
plt.figlegend(loc='lower center', fontsize=25, ncol=3, bbox_to_anchor=(0.5, -0.2))

# Show
plt.savefig(f"../figures_first_hateful/Sensitivity_Analysis_Falsification_{SUBREDDIT}_{LABEL}.png", bbox_inches='tight')
plt.show()


# ## Interrupted Time Series

# ### Prepare Data

# In[17]:


# Specify the date period under which to calculate means
bandwidth= RMSEs.T[0].idxmin()

# Keep only data within the desired date range
df_treatment_filtered = df_treatment[(df_treatment.index >= -bandwidth) & (df_treatment.index < bandwidth)].copy()
df_control_filtered = df_control[(df_control.index >= -bandwidth) & (df_control.index < bandwidth)].copy()

# Add class labels
df_treatment_filtered['exposed'] = 1
df_control_filtered['exposed'] = 0

# Split before and after
df_treatment_before = df_treatment_filtered[df_treatment_filtered.index < 0].copy()
df_treatment_after = df_treatment_filtered[df_treatment_filtered.index >= 0].copy()
df_control_before = df_control_filtered[df_control_filtered.index < 0].copy()
df_control_after = df_control_filtered[df_control_filtered.index >= 0].copy()
df_treatment_before['interrupted'] = 0
df_treatment_after['interrupted'] = 1
df_control_before['interrupted'] = 0
df_control_after['interrupted'] = 1

# Merge 
df = pd.concat([df_treatment_before, df_treatment_after, 
                df_control_before, df_control_after])

# Then convert the index to a feature (time)
df.reset_index(inplace=True)
df.columns = ['time', 'percentage', 'exposed', 'interrupted']
df['time_x_exposed'] = df['time'] * df['exposed']
df['time_x_interrupted'] = df['time'] * df['interrupted']
df['exposed_x_interrupted'] = df['exposed'] * df['interrupted']
df['time_x_exposed_x_interrupted'] = df['time'] * df['exposed'] * df['interrupted']


# ### Regression

# In[18]:


# Split X and Y
x = df.drop(columns='percentage')
y = df['percentage']

# OLS Regression Results 
model = sm.OLS(endog=y, exog=sm.add_constant(x)).fit()
results = model.summary()
print(results)


# In[19]:


# Calculate error bars
predstd, error_lower, error_upper = wls_prediction_std(model)

errors = x.copy()
errors['error_lower'] = error_lower
errors['error_upper'] = error_upper

errors.to_csv(f'../plot_data_first_hateful/Error_Intervals_{SUBREDDIT}_{LABEL}.csv', index=False)


# ### Generate Regression Lines

# In[20]:


# Get X
X_treatment_before = df[(df.exposed == 1) & (df.time <= 0)].drop(columns='percentage')
X_treatment_after = df[(df.exposed == 1) & (df.time >= 0)].drop(columns='percentage')
X_control_before = df[(df.exposed == 0) & (df.time <= 0)].drop(columns='percentage')
X_control_after = df[(df.exposed == 0) & (df.time >= 0)].drop(columns='percentage')

# Adjust 'interrupted' label for pre-treatment features at 0
X_treatment_before[[col for col in X_treatment_before.columns if 'interrupted' in col]] = 0
X_control_before[[col for col in X_control_before.columns if 'interrupted' in col]] = 0

# Predict to get Y
y_treatment_before = model.predict(sm.add_constant(X_treatment_before, has_constant='add'))
y_treatment_after = model.predict(sm.add_constant(X_treatment_after, has_constant='add'))
y_control_before = model.predict(sm.add_constant(X_control_before, has_constant='add'))
y_control_after = model.predict(sm.add_constant(X_control_after, has_constant='add'))

# Change indexes on predition to relative times
y_treatment_before.index = X_treatment_before.time
y_treatment_after.index = X_treatment_after.time
y_control_before.index = X_control_before.time
y_control_after.index = X_control_after.time


# In[21]:


# Save info for plotting outside
df.to_csv(f'../plot_data_first_hateful/ITS_Plot_Data_{SUBREDDIT}_{LABEL}_df.csv', index=False)
y_treatment_before.to_csv(f'../plot_data_first_hateful/ITS_Plot_Data_{SUBREDDIT}_{LABEL}_y_treatment_before.csv', index=False)
y_treatment_after.to_csv(f'../plot_data_first_hateful/ITS_Plot_Data_{SUBREDDIT}_{LABEL}_y_treatment_after.csv', index=False)
y_control_before.to_csv(f'../plot_data_first_hateful/ITS_Plot_Data_{SUBREDDIT}_{LABEL}_y_control_before.csv', index=False)
y_control_after.to_csv(f'../plot_data_first_hateful/ITS_Plot_Data_{SUBREDDIT}_{LABEL}_y_control_after.csv', index=False)


# In[22]:


# Get X and Y
x = df.drop(columns='percentage')
y = df['percentage']

# Create figure
fig, ax = plt.subplots(figsize=(16,9))

# Aggregate data weekly and plot error bars
df_weekly = df.copy()
df_weekly['time'] = df_weekly['time']//7
plt.errorbar(x=df_weekly[df_weekly.exposed == 1].groupby('time')['time'].mean(), 
             y=df_weekly[df_weekly.exposed == 1].groupby('time')['percentage'].mean(), 
             yerr=df_weekly[df_weekly.exposed == 1].groupby('time')['percentage'].sem(),
             fmt='o', capsize=4,
             lw=2, color='C1', alpha=0.40, label=f'Treatment Users Outside')
plt.errorbar(x=df_weekly[df_weekly.exposed == 0].groupby('time')['time'].mean(), 
             y=df_weekly[df_weekly.exposed == 0].groupby('time')['percentage'].mean(), 
             yerr=df_weekly[df_weekly.exposed == 0].groupby('time')['percentage'].sem(),
             fmt='o', capsize=4,
             lw=2, color='C4', alpha=0.40, label=f'Control Users')

# Add Regression Lines
y_treatment_before.index = X_treatment_before.time // 7
y_treatment_after.index = X_treatment_after.time // 7
y_control_before.index = X_control_before.time // 7
y_control_after.index = X_control_after.time // 7
plt.plot([y_treatment_before.idxmin(), y_treatment_before.idxmax()], [y_treatment_before.min(), y_treatment_before.max()],
         color='C1', lw=4)
plt.plot([y_treatment_after.idxmin(), y_treatment_after.idxmax()], [y_treatment_after.min(), y_treatment_after.max()],
         color='C1', lw=4)
plt.plot([y_control_before.idxmin(), y_control_before.idxmax()], [y_control_before.min(), y_control_before.max()],
         color='C4', lw=4)
plt.plot([y_control_after.idxmin(), y_control_after.idxmax()], [y_control_after.min(), y_control_after.max()],
         color='C4', lw=4)

# Add join date
plt.axvline(x=0, color='red', ymin=0.025, ymax=0.975, alpha=0.75, linestyle='--', linewidth=4, label=f'User Became Active')

# Text
plt.legend(fontsize=30, frameon=False, loc='upper left') #, bbox_to_anchor=(0.4, 0))
plt.title(SUBREDDIT, size=40, pad=20)
plt.xlabel('Week', fontsize=40, weight='normal', labelpad=20)
plt.ylabel('Hate Speech', fontsize=40, weight='normal', labelpad=20)

# Axis
#plt.ylim([0,0.01])
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
#ax.yaxis.set_major_locator(mtick.MultipleLocator(0.002))
ax.locator_params(axis ='y', nbins=6)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=2))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show
plt.savefig(f"../figures_first_hateful/Interrupted_Time_Series_{SUBREDDIT}_{LABEL}.png", bbox_inches='tight')
plt.show()


# ### Residuals

# In[23]:


# Calculate residuals
df_res = df.copy()
df_res['residuals'] = model.resid
residuals_line = sm.OLS(endog=df_res.residuals, exog=sm.add_constant(df_res.time)).fit()

# Create figure
fig, ax = plt.subplots(figsize=(12,6))

# Residuals
plt.errorbar(x=df_res.groupby('time')['time'].mean(), 
             y=df_res.groupby('time')['residuals'].mean(), 
             yerr=df_res.groupby('time')['residuals'].sem(),
             color='C2', lw=2, fmt='o', capsize=4, alpha=0.75)
#plt.plot(df_res['time'].unique(), df_res['time'].unique() * residuals_line.params['time'], color='magenta')

# Text
plt.title(SUBREDDIT, size=30, pad=20)
plt.xlabel('Day', fontsize=30, weight='normal', labelpad=20)
plt.ylabel('Residuals', fontsize=30, weight='normal', labelpad=20)

# Axis
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.locator_params(axis ='y', nbins=6)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show
plt.savefig(f"../figures_first_hateful/Residuals_{SUBREDDIT}_{LABEL}.png", bbox_inches='tight')
plt.show()


# ## Daily Deltas

# In[24]:


X_treatment_before = pd.DataFrame()
X_treatment_after = pd.DataFrame()
X_control_before = pd.DataFrame()
X_control_after = pd.DataFrame()

# Before
for time in range(-bandwidth, 0):
    
    X_treatment_before = X_treatment_before.append({
        'time': time,
        'exposed': 1,
        'interrupted': 0,
        'time_x_exposed': time,
        'time_x_interrupted': 0,
        'exposed_x_interrupted': 0,
        'time_x_exposed_x_interrupted': 0
    }, ignore_index=True)
    
    X_control_before = X_control_before.append({
        'time': time,
        'exposed': 0,
        'interrupted': 0,
        'time_x_exposed': 0,
        'time_x_interrupted': 0,
        'exposed_x_interrupted': 0,
        'time_x_exposed_x_interrupted': 0
    }, ignore_index=True)

    
# After
for time in range(0, bandwidth):
    
    X_treatment_after = X_treatment_after.append({
        'time': time,
        'exposed': 1,
        'interrupted': 1,
        'time_x_exposed': time,
        'time_x_interrupted': time,
        'exposed_x_interrupted': 1,
        'time_x_exposed_x_interrupted': time
    }, ignore_index=True)
    
    X_control_after = X_control_after.append({
        'time': time,
        'exposed': 0,
        'interrupted': 1,
        'time_x_exposed': 0,
        'time_x_interrupted': time,
        'exposed_x_interrupted': 0,
        'time_x_exposed_x_interrupted': 0
    }, ignore_index=True)


# In[25]:


# Generate prediction lines
y_treatment_before = model.predict(sm.add_constant(X_treatment_before, has_constant='add'))
y_treatment_after = model.predict(sm.add_constant(X_treatment_after, has_constant='add'))
y_control_before = model.predict(sm.add_constant(X_control_before, has_constant='add'))
y_control_after = model.predict(sm.add_constant(X_control_after, has_constant='add'))

# Calculate deltas
delta_before = y_treatment_before - y_control_before
delta_after = y_treatment_after - y_control_after

# Update index
delta_before.index = X_treatment_before.time.astype(int)
delta_after.index = X_treatment_after.time.astype(int)
delta_before.name = 'Hate_Speech_Delta'
delta_after.name = 'Hate_Speech_Delta'

# Save for plots
delta_before.to_csv(f"../effect_deltas/Effect_Delta_Before_{SUBREDDIT}_{LABEL}.csv")
delta_after.to_csv(f"../effect_deltas/Effect_Delta_After_{SUBREDDIT}_{LABEL}.csv")


# In[26]:


# Create figure
fig, ax = plt.subplots(figsize=(15,5))

# Plots
ax.plot(delta_before, lw=4, color='darkslategray')
ax.plot(delta_after, lw=4, color='darkslategray')

# Add join date
plt.axvline(x=0, color='red', ymin=0.025, ymax=0.975, alpha=0.75, label=f'User Joined r/{SUBREDDIT}')

# Text
plt.title('Predicted Excess Hate Speech for Treatment Group', size=30, pad=20)
plt.xlabel('Day', fontsize=30, weight='normal', labelpad=20)
plt.ylabel('Hate Speech', fontsize=30, weight='normal', labelpad=20)
plt.legend(fontsize=20)

# Axis
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.locator_params(axis ='y', nbins=6)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=2))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show
plt.savefig(f"../figures_first_hateful/Daily_Deltas_{SUBREDDIT}_{LABEL}.png", bbox_inches='tight')
plt.show()


# # End
