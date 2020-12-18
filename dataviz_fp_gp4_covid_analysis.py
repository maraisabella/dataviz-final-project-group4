#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[2]:


os.getcwd()


# In[6]:


df = pd.read_csv('owid-covid-data.csv')


# In[4]:


usa_df = df[df.location=='United States']
usa_df.columns


# In[9]:


import seaborn as sns
sns.heatmap(df.isnull())


# In[10]:


sns.heatmap(usa_df.isnull())


# In[13]:


df.isnull().sum()


# In[19]:


cols = ['reproduction_rate', 'new_cases','new_tests', 'new_deaths', 'icu_patients', 
        'positive_rate', 'stringency_index', 'population', 'population_density', 
        'median_age', 'aged_65_older', 'aged_70_older','gdp_per_capita','extreme_poverty', 
        'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 
        'handwashing_facilities','hospital_beds_per_thousand', 'life_expectancy', 'human_development_index'] 


# In[20]:


new_df =  df[cols]


# In[21]:


new_df.head()


# In[27]:


fig = pd.plotting.scatter_matrix(new_df, alpha=0.2)
plt.savefig('scatterPlot.png')


# In[ ]:




