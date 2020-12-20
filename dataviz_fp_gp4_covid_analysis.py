
###############################################################################################
## dataviz_fp_gp4_covid_analysis.py                                                          ##
##                                                                                           ##
##                                                                                           ##
## Created by: Group4 Data Visualization Boot Camp Team                                      ##
###############################################################################################                                   


import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###############################################################################################
## This code implements the DataBase logic                                                   ##
##                                                                                           ##
###############################################################################################                                   
from sqlalchemy import create_engine

# create engine
print(">>>>>>>>>> Connecting to covid.sqlite database <<<<<<<<<<")
engine = create_engine("sqlite:///database/covid.sqlite")
conn = engine.connect()

# Query to select all rows from owid_covid_data table. 
data_df = pd.read_sql("SELECT * FROM owid_covid_data", conn)

# print first 5 rows of data_df dataframe
print (data_df.head)

###############################################################################################
## This code implements the Data Cleaning                                                   ##
##                                                                                           ##
###############################################################################################
# Check total null values for each column
print(data_df.isnull().sum())
data_df.drop(['tests_per_case',
'new_tests',
'total_tests',
'positive_rate',
'handwashing_facilities',
'tests_units',
'new_tests_smoothed',
'new_tests_smoothed_per_thousand',
'new_tests_per_thousand',
'total_tests_per_thousand',
'weekly_hosp_admissions_per_million',
'weekly_icu_admissions_per_million',
'new_cases_per_million',
'total_cases_per_million',
'iso_code','continent',
'new_cases_smoothed',
'new_deaths_smoothed',
'total_deaths',
'total_deaths',
'new_deaths',
'total_deaths_per_million',
'new_cases_smoothed_per_million',
'new_deaths_per_million',
'new_deaths_smoothed_per_million',
'icu_patients',
'icu_patients_per_million',
'hosp_patients_per_million',
'weekly_icu_admissions',
'hosp_patients',
'total_vaccinations',
'total_vaccinations_per_hundred',
'weekly_hosp_admissions'],axis=1, inplace=True)

cleaned_df = data_df.dropna()

print(cleaned_df.head)

cleaned_df.to_sql('cleaned_data', con=engine, if_exists='replace')

###############################################################################################
## This code implements the Machine Learning model                                           ##
## Author: Muhammad Ovais Naeem                                                                                        ##
###############################################################################################                                   

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Connect to cleaned_data database
data_to_model_df = pd.read_sql("SELECT * FROM cleaned_data", conn)

# select features for ML model
features = [
 'total_cases',
 'reproduction_rate',
 'stringency_index',
 'population',
 'population_density',
 'median_age',
 'aged_65_older',
 'aged_70_older',
 'gdp_per_capita',
 'extreme_poverty',
 'cardiovasc_death_rate',
 'diabetes_prevalence',
 'female_smokers',
 'male_smokers',
 'hospital_beds_per_thousand',
 'life_expectancy',
 'human_development_index']

X = data_to_model_df[features]

# X = cleaned_df['total_cases'].values.reshape(-1, 1)

# The dependent variable
y = data_to_model_df['new_cases']

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Instantiate linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print('Intercept: \n', model.intercept_)
print('Coefficients: \n', model.coef_)

# Create predictions
y_pred = model.predict(X_test)

mean_sq_error = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print(f"Mean Square Error for this model is: {mean_sq_error}")
print(f"R squared value for this model is: {r2_score}")
# plt.scatter(X, y)
# plt.plot(X, y_pred, c='red')
# plt.savefig("linear_regression.png")


