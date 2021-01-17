# dataviz-final-project-group4
Final project for the McCombs Data and Visualization Bootcamp

## Project
### Topic: Prediction of early re-admission for hospitalized patients with diabetic care.
### Reason for topic selection:
Availability of clinical data containing valuable information.

### Description of the source of data: https://www.hindawi.com/journals/bmri/2014/781670/#supplementary-materials
### Questions hoping to be answered with the data: 
- To analyze the clinical data and predict early re-admission (within 30 days) of diabetic patients.

### Description of the communication protocols: 
- In order to enhance our communication in the most effective manner, we will be communicating via Slack for project updates. We will also be conductive team Zoom meetings every day. With both of these methods being used for our project, they are both equally crucial because we will be notifying each other of changes made to code and coordinating with each other before changes are pushed, pulled, and merged on GitHub.

## Overview
The purpose of this project is to implement end to end data pipeline and finally analyze and model the data using ML techniques.

## Exploratory Data Analysis


## Machine Learning Model
Since this is a classification problem, we started with Logistic Regression model to predict the binary outcome of whether the patient will be re-admitted or not.

## Data Pre-Processing
Preliminary data pre-processing involves two steps:
- Label encoding
- Dropping original columns
- Data scaling

## Preliminary Feature Engineering
- For preliminary feature engineering we used the coeff_ property of LogisticRegression model that shows the coefficients found for each input variable:

![](analysis/feature_imp.png) 

## Description of Data Splitting
- For data splitting we used the 'shape' method to identify the percentage of 'train' set and 'test' set. Following is the code snippet:
![](analysis/Test_Train_Set_Percentage.png)

Our cleaned dataframe contains a total of 69710 rows. Doing simple math we can see that there is a 75-25 split between train set and test set respectively.

The provisional machine learning model we have used is Logistic Regression to predict patient re-admission based on the following features:

features = ['race',
 'gender',
 'age',
 'medical_specialty',
 'diag_1',
 'diag_2',
 'diag_3',
 'max_glu_serum',
 'A1Cresult',
 'metformin',
 'repaglinide',
 'nateglinide',
 'chlorpropamide',
 'glimepiride',
 'acetohexamide',
 'glipizide',
 'glyburide',
 'tolbutamide',
 'pioglitazone',
 'rosiglitazone',
 'acarbose',
 'miglitol',
 'troglitazone',
 'tolazamide',
 'examide',
 'citoglipton',
 'insulin',
 'glyburide-metformin',
 'glipizide-metformin',
 'glimepiride-pioglitazone',
 'metformin-rosiglitazone',
 'metformin-pioglitazone',
 'change',
 'diabetesMed']

#### Screenshot of ML code:


### Database Integration

- **pgAdmin/PostgreSQL** is the database we used for segment 2 analysis.

- We are connecting to the postgresSQL database and accessing the data in the python file using the "psycopg2" and "sqlalchemy".

- **'diabetes_raw_data'** table is created in the postgresSQL database to hold all the raw data. The raw data is imported from the csv file - diabetic_data_initial.csv.

- The raw data from the postgresSQL table 'diabetes_raw_data' is read into dataframe - diabetes_raw_data_df.

- After the data cleaning procedure is completed, the cleaned data in the dataframe diabetes_raw_data_df is written to a new table **'diabetes_clean_data'**. The new table 'diabetes_clean_data' is created while copying the data from dataframe into sql.

- Multiple other tables are created from 'diabetes_clean_data' table in the database.sql file.

- The tables are : Patient, Admission, Diagnosis and Medicines.

- Following is the snapshots of **ERD** used:

    ![](database/DiabetesERD.png) 

- **Schema** can be found in *database/diabetes_schema.txt*.

- Inner join is performed from Patient into the Admission table to view the pateints who are re-admitted.

### Scripts
Following are the details of scripts and supporting files for this project:
- diabetes_dataset_cleaning_merged.ipynb - Final Jupyter Notebook file used for data cleaning and exploratory data analysis.
- dataviz_fp_gp4_core.py - Core Python file to perform end to end process of data cleaning, connecting to database and ML algorithm.


### Data Visualization
- The dashboard will be demonstrated using Google slides. The following link directs to the presentation. https://docs.google.com/presentation/d/1_bkTXbazjyO1s4JLAkCp77HT9BQO3BVKlw5PU5_HBhI/edit?usp=sharing
- Tableau will be used as primary tool for data visualization and dashboard creation
- Various plots/stats were generated to analyze the impact on hospital re-admission 
- Plots and visualizations are mainly static in Tableau, however filters will be provided to visualize the plots/stats based on parameters of interest.
- Following is a blueprint of the storyboard that will be improved and have more content added to in the next project deliverables:

![](analysis/dashboard.png)






