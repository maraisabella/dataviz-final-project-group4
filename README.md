# dataviz-final-project-group4
Final project for the McCombs Data and Visualization Bootcamp

## Project
### Topic: COVID-19 Data Analysis/Modeling to Predict New Cases across countries
### Reason for topic selection: 
### Description of the source of data: https://github.com/owid/covid-19-data/tree/master/public/data
### Questions hoping to be answered with the data: 
- To analyze impact of covid on economy (some relationship between covid cases vs GDP)
- To analyze relationship between underlying health conditions and deaths
- Predict new cases

### Description of the communication protocols: 
Slack, Team meetings every other day.

## Overview

### Exploratory Data Analysis

### Machine Learning Model
The provisional machine learning model we have used is Linear Regression to predict new Covid cases.

### Database Integration
SQLite is the database we intend to use for the initial analysis.

We intend to use Python, Pandas functions and methods, and SQLAlchemy to filter out the columns in the table from the covid.sqlite database to retrive all the data for each country.

SQLite is selected in the initial phase of the project as it is easy to setup. It is a file-based relational database that uses SQL as its query language. Being file-based tremendously simplifies deployment, making it very good for the case where an application needs a little database but must be run in an environment where having a database server would be problematic.

We created a SQLite dabase "covid.sqlite" in the folder database. This database is created by importing the cleaned data from the CSV file "owid-covid-data.csv".

### Data Visualization





