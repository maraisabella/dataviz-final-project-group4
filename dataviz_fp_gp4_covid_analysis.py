#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import matplotlib.pyplot as plt


os.getcwd()

df = pd.read_csv('owid-covid-data.csv')

usa_df = df[df.location=='United States']
usa_df.columns

import seaborn as sns
sns.heatmap(df.isnull())

sns.heatmap(usa_df.isnull())

df.isnull().sum()

cols = ['reproduction_rate', 'new_cases','new_tests', 'new_deaths', 'icu_patients', 
        'positive_rate', 'stringency_index', 'population', 'population_density', 
        'median_age', 'aged_65_older', 'aged_70_older','gdp_per_capita','extreme_poverty', 
        'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 
        'handwashing_facilities','hospital_beds_per_thousand', 'life_expectancy', 'human_development_index'] 

new_df =  df[cols]

new_df.head()

fig = plt.figure()
pd.plotting.scatter_matrix(new_df, alpha=0.2)
fig.savefig('scatter_plot_matrix.png', dpi=fig.dpi)
