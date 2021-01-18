###############################################################################################
## dataviz_fp_gp4_core.py                                                                    ##
##                                                                                           ##
##                                                                                           ##
## Created by: Group4 Data Visualization Boot Camp Team                                      ##
############################################################################################### 

# Import dependencies 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import collections
import researchpy as rp
import scipy as sp
from sklearn.preprocessing import LabelEncoder
import re
import os
import matplotlib.pyplot as plt

###############################################################################################
## This code implements the DataBase logic                                                   ##
##                                                                                           ##
############################################################################################### 
# Database Dependancies
import sqlalchemy
from sqlalchemy import create_engine
import psycopg2
from config import db_password
from db_params import database_port, csv_path

# Database Credentials
DB_HOST = "127.0.0.1"
DB_PORT = database_port
DB_NAME = "DiabeticDB"
DB_USER = "postgres"
DB_PASS = db_password
CSV_FILE_PATH = csv_path

print(">>>>>>>>>> Connecting to DiabeticDB database <<<<<<<<<<")
conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=db_password, host=DB_HOST, port=DB_PORT)
print("Database opened successfully")

# Create a cursor
cur = conn.cursor()

# Drop the diabetes_raw_data if exists before creating one
cur.execute("DROP TABLE IF EXISTS medicines");
cur.execute("DROP TABLE IF EXISTS admission");
cur.execute("DROP TABLE IF EXISTS diagnosis");
cur.execute("DROP TABLE IF EXISTS patient");
cur.execute("DROP TABLE IF EXISTS diabetes_clean_data");
cur.execute("DROP TABLE IF EXISTS diabetes_raw_data");
conn.commit()

# Create diabetes_raw_data table
cur = conn.cursor()
cur.execute('''CREATE TABLE diabetes_raw_data (
    encounter_id int NOT NULL,
    patient_nbr	int NOT NULL,
    race varchar(20),
    gender varchar(20) NOT NULL,
    age varchar(10) NOT NULL,
    weight varchar(10),
    admission_type_id int NOT NULL,
    discharge_disposition_id int NOT NULL,
    admission_source_id int NOT NULL,
    time_in_hospital int NOT NULL,
    payer_code varchar(10),
    medical_specialty varchar(40),
    num_lab_procedures  int NOT NULL,
    num_procedures  int NOT NULL,
    num_medications  int NOT NULL,
    number_outpatient  int NOT NULL,
    number_emergency  int NOT NULL,
    number_inpatient  int NOT NULL,
    diag_1  varchar(10),
    diag_2 varchar(10),
    diag_3 varchar(10),
    number_diagnoses  int NOT NULL,
    max_glu_serum  varchar(10),
    A1Cresult  varchar(10),
    metformin  varchar(10),
    repaglinide  varchar(10),
    nateglinide  varchar(10),
    chlorpropamide  varchar(10),
    glimepiride  varchar(10),
    acetohexamide  varchar(10),
    glipizide  varchar(10),
    glyburide  varchar(10),
    tolbutamide  varchar(10),
    pioglitazone  varchar(10),
    rosiglitazone  varchar(10),
    acarbose  varchar(10),
    miglitol  varchar(10),
    troglitazone  varchar(10),
    tolazamide  varchar(10),
    examide  varchar(10),
    citoglipton  varchar(10),
    insulin  varchar(10),
    "glyburide-metformin"  varchar(10),
    "glipizide-metformin"  varchar(10),
    "glimepiride-pioglitazone"  varchar(10),
    "metformin-rosiglitazone"  varchar(10),
    "metformin-pioglitazone"  varchar(10),
    change  varchar(10),
    diabetesMed  varchar(10),
    readmitted  varchar(10),
    PRIMARY KEY (encounter_id),
    UNIQUE (encounter_id));''')
print("Table 'diabetes_raw_data' successfully created")

# Copy the contents from the diabetic_data_initial.csv and write it to 'diabetes_raw_data' table
cur = conn.cursor()
cmd = f'''COPY diabetes_raw_data FROM '{CSV_FILE_PATH}' CSV HEADER DELIMITER ',';'''
# cur.execute('''COPY diabetes_raw_data
#     FROM \'CSV_FILE_PATH\'
#     CSV HEADER DELIMITER ',';''')
cur.execute(cmd)

###############################################################################################
## This code implements the Data Cleaning Logic                                              ##
##                                                                                           ##
###############################################################################################

# Create the connection to the PostgreSQL database
db_string = f"postgres://{DB_USER}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_string)
conn.commit()

df = pd.read_sql_table("diabetes_raw_data", engine)
print(df.head)

pd.set_option('display.max_columns', None)
print(df.head)

# Show columns with missing greater than 20%
for column in df:
    if df[df[column]=='?'].shape[0]/df.shape[0]*100 > 20:
        print(column,":",str(round(df[df[column]=='?'].shape[0]/df.shape[0]*100)) + "%")  

# variables weight and payer_code were excluded due to quantity of missing data
# medical specialty was recoded to add "missing" for the missing values.

# drop weight and payer_code columns 
df_copy = df.copy()
df_copy.drop(columns=['weight', 'payer_code'], axis=1, inplace=True)

# In order to keep observations independent, only the first encounter is included
# Dedupe based on first encounter
df_deduped = df_copy.drop_duplicates(subset=['patient_nbr'], keep='first')

# Remove encounters that resulted in either discharge to 
# a hospice or patient death to avoid biasing analysis 
discharge_disposition_excluded=[11, 13, 14, 19, 20, 23]

df_cleaned = df_deduped[~df_deduped.discharge_disposition_id.isin(discharge_disposition_excluded)]

print(df_cleaned['gender'].value_counts())

df_cleaned.drop(df_cleaned[(df_cleaned.loc[:,'gender'] == "Unknown/Invalid")].index, inplace = True)
print("Dropping unknown/invalid gender entries from dataframe")
print(df_cleaned['gender'].value_counts())

# recode readmitted to be binary 
df_cleaned['readmitted_recoded'] = df_cleaned.loc[:,['readmitted']].replace({'>30': 'NO'})

# recode medical_specialty to add missing 
df_cleaned['medical_specialty_recoded'] = df_cleaned.loc[:,['medical_specialty']].replace("?",'missing', inplace=False)

# readmitted recoded is imbalanced with 91% of cases 
# not having a 30 day readmission
print(collections.Counter(df_cleaned['readmitted_recoded']))

print(df_cleaned['medical_specialty_recoded'].value_counts())

df_cleaned['medical_specialty_recoded'] = df_cleaned.loc[:,['medical_specialty_recoded']].replace({'Family/GeneralPractice': 'InternalMedicine'})

# Function to replace medical specialties with "Other" except "Internal Medicine"
def values_to_other(col_name, value_unchanged):
    v=[]
    for value in col_name:
        if (value != value_unchanged):
            v.append("Other")
        else:
            v.append(value)
    return v
    
# Apply values_to_other function to the "medical_specialty_recoded" column
df_cleaned['medical_specialty_recoded'] = values_to_other(df_cleaned['medical_specialty_recoded'].values,"InternalMedicine")    
print(df_cleaned['medical_specialty_recoded'].value_counts())

# readmitted recoded is imbalanced with 91% of cases 
# not having a 30 day readmission
print(collections.Counter(df_cleaned['readmitted_recoded']))

# function to clean 'age' column
def parse_age_range(age_col):
    c=[]
    for values in age_col:
        s = re.sub('[[)]','', values)
        c.append(s)
    return c

# Clean "age" column from brackets
df_cleaned['age'] = parse_age_range(df_cleaned['age'].values)

# Drop columns "readmitted" and "medical_specialty"
df_cleaned = df_cleaned.drop(columns=['readmitted', 'medical_specialty'])

# DB code to export cleaned dataset to DB
# Create a cursor
cur = conn.cursor()
# Drop the diabetes_cleaned_data if exists before creating one
cur.execute("DROP TABLE IF EXISTS diabetes_clean_data");
conn.commit()

# Write the cleaned data fro diabetes_clean_data_df into postgres diabetes_clean_data table 
df_cleaned.to_sql(name='diabetes_clean_data', con=engine, if_exists='replace', index=False,
            dtype={'encounter_id': sqlalchemy.types.INTEGER(),
                   'patient_nbr' : sqlalchemy.types.INTEGER(),
                   'race' : sqlalchemy.types.VARCHAR(length=20),
                   'gender' : sqlalchemy.types.VARCHAR(length=20),
                   'age' : sqlalchemy.types.VARCHAR(length=10),
                   'admission_type_id' : sqlalchemy.types.INTEGER(),
                   'discharge_disposition_id' : sqlalchemy.types.INTEGER(),
                   'admission_source_id' : sqlalchemy.types.INTEGER(),
                   'time_in_hospital' : sqlalchemy.types.INTEGER(),
                   'num_lab_procedures' :  sqlalchemy.types.INTEGER(),
                   'num_procedures' :  sqlalchemy.types.INTEGER(),
                   'num_medications' :  sqlalchemy.types.INTEGER(),
                   'number_outpatient' :  sqlalchemy.types.INTEGER(),
                   'number_emergency' :  sqlalchemy.types.INTEGER(),
                   'number_inpatient' :  sqlalchemy.types.INTEGER(),
                   'diag_1' :  sqlalchemy.types.VARCHAR(length=10),
                   'diag_2' : sqlalchemy.types.VARCHAR(length=10),
                   'diag_3' : sqlalchemy.types.VARCHAR(length=10),
                   'number_diagnoses' :  sqlalchemy.types.INTEGER(),
                   'max_glu_serum' :  sqlalchemy.types.VARCHAR(length=10),
                   'A1Cresult' :  sqlalchemy.types.VARCHAR(length=10),
                   'metformin' :  sqlalchemy.types.VARCHAR(length=10),
                   'repaglinide' :  sqlalchemy.types.VARCHAR(length=10),
                   'nateglinide' :  sqlalchemy.types.VARCHAR(length=10),
                   'chlorpropamide' :  sqlalchemy.types.VARCHAR(length=10),
                   'glimepiride' :  sqlalchemy.types.VARCHAR(length=10),
                   'acetohexamide' :  sqlalchemy.types.VARCHAR(length=10),
                   'glipizide' :  sqlalchemy.types.VARCHAR(length=10),
                   'glyburide' :  sqlalchemy.types.VARCHAR(length=10),
                   'tolbutamide' :  sqlalchemy.types.VARCHAR(length=10),
                   'pioglitazone' :  sqlalchemy.types.VARCHAR(length=10),
                   'rosiglitazone' :  sqlalchemy.types.VARCHAR(length=10),
                   'acarbose' :  sqlalchemy.types.VARCHAR(length=10),
                   'miglitol' :  sqlalchemy.types.VARCHAR(length=10),
                   'troglitazone' :  sqlalchemy.types.VARCHAR(length=10),
                   'tolazamide' :  sqlalchemy.types.VARCHAR(length=10),
                   'examide' :  sqlalchemy.types.VARCHAR(length=10),
                   'citoglipton' :  sqlalchemy.types.VARCHAR(length=10),
                   'insulin' :  sqlalchemy.types.VARCHAR(length=10),
                   'glyburide-metformin' :  sqlalchemy.types.VARCHAR(length=10),
                   'glipizide-metformin' :  sqlalchemy.types.VARCHAR(length=10),
                   'glimepiride-pioglitazone' : sqlalchemy.types.VARCHAR(length=10),
                   'metformin-rosiglitazone' :  sqlalchemy.types.VARCHAR(length=10),
                   'metformin-pioglitazone' :  sqlalchemy.types.VARCHAR(length=10),
                   'change' : sqlalchemy.types.VARCHAR(length=10),
                   'diabetesMed' :  sqlalchemy.types.VARCHAR(length=10),
                   'readmitted_recoded' :  sqlalchemy.types.VARCHAR(length=10),
                   'medical_specialty_recoded' : sqlalchemy.types.VARCHAR(length=40)})
print("Table 'diabetes_clean_data' successfully created")

# Setting the encounter_id as 'PRIMARY KEY'
with engine.connect() as con:
    con.execute('ALTER TABLE diabetes_clean_data ADD PRIMARY KEY (encounter_id);')
    conn.commit()
    
###############################################################################################
## This code implements the Exploratory Data Analysis                                        ##
##                                                                                           ##
###############################################################################################                                   


###############################################################################################
## This code implements the Pre-Processing for Machine Learning model                        ##
##                                                                                           ##
###############################################################################################                                   

# Read the clean data from the postgres into dataframe
df_cleaned = pd.read_sql_table("diabetes_clean_data", engine)
print(df_cleaned.head)

# Create a list of columns to encode for each variable if variable type is object
columns_to_encode = [column for column in df_cleaned.columns if df_cleaned[column].dtypes == 'O']

print(columns_to_encode)

# exclude diag 1, 2, and 3 from recode list 
for i in range(3):
    to_remove = 'diag_' + str(i+1)
    columns_to_encode.remove(to_remove)
print(columns_to_encode)
print("\n", len(columns_to_encode), "columns")

# Make a copy of cleaned dataframe for processing through ML model
encoded_df = df_cleaned.copy()

# function to apply label encoding
def apply_encoder(cols):
    le = LabelEncoder()
    for c in cols:
        new_column_name = c + "_le"
        le.fit(df_cleaned[c])
        encoded_df[new_column_name] = le.transform(encoded_df[c])
        print(c, ":", le.classes_)

# Apply Encoding
apply_encoder(columns_to_encode)

print(f"Column names after encoding: \n{encoded_df.columns}")

cols_to_drop = ['race', 'gender', 'age',
       'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
       'time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
       'max_glu_serum', 'a1cresult', 'metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
       'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
       'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
       'insulin', 'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesmed', 'readmitted_recoded',
       'medical_specialty_recoded']
encoded_df = encoded_df.drop(columns=cols_to_drop)

# Set ? to NaN
encoded_df.replace('?', np.nan, inplace=True)

# Show missing in dataset 
for column in df_cleaned.columns:
    missing_count = df_cleaned[column].isnull().sum()
    if missing_count>0:
        print(column,":",df_cleaned)   
        
 
###############################################################################################
## To be incorporated later                                                                  ##
##                                                                                           ##
###############################################################################################                                   
       

# # check multivariate outliers 
# # credit to https://www.machinelearningplus.com/statistics/mahalanobis-distance/
# def mahalanobis(x=None, data=None, cov=None):
#     """Compute the Mahalanobis Distance between each row of x and the data  
#     x    : vector or matrix of data with, say, p columns.
#     data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
#     cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
#     """
#     x_minus_mu = x - np.mean(data)
#     if not cov:
#         cov = np.cov(data.values.T)# covarience of data.values transposed 
#     inv_covmat = sp.linalg.inv(cov)
#     left_term = np.dot(x_minus_mu, inv_covmat)
#     mahal = np.dot(left_term, x_minus_mu.T)
#     return mahal.diagonal()        
        
# #create new column in dataframe that contains Mahalanobis distance for each row
# df_x = df_cleaned[['time_in_hospital', 'num_lab_procedures', 'num_procedures',
#        'num_medications', 'number_outpatient', 'number_emergency','number_diagnoses']]
# print(df_x.head())        

# # df_x = df[['carat', 'depth', 'price']].head(500)
# df_cleaned['mahala'] = mahalanobis(x=df_x, data=df_cleaned[['time_in_hospital', 'num_lab_procedures', 'num_procedures',
#        'num_medications', 'number_outpatient', 'number_emergency','number_diagnoses']])
# print(df_cleaned.head())

# # Check multicollinearity 
# # credit https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/
# # Import library for VIF - varience inflation factor 
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# def calc_vif(X):
#     # Calculating VIF
#     vif = pd.DataFrame()
#     vif["variables"] = X.columns
#     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#     return(vif)

# # variables with high VIF might be removed one at a time in descending order
# calc_vif(df_cleaned.iloc[:,:-1])

###############################################################################################
## Summary Statistics: EDA                                                                   ##
##                                                                                           ##
###############################################################################################                                   

# 69,710 records in the dataset 
display(df_cleaned.describe())

# 69,710 records in the dataset 
display(df_cleaned.describe())

# 53.25% of admissions came from Trauma Center followed by ED
display(rp.summary_cat(df_cleaned[['admission_source_id']]))

# ~75% of cases were caucasian followed by 18% for African American 
display(rp.summary_cat(df_cleaned[['race']]))

# 53% cases were female, 47% male 
display(rp.summary_cat(df_cleaned[['gender']]))
df_cleaned.gender.value_counts().plot(kind='bar', title='gender')
plt.show()

#~81% of cases above age 50
df_cleaned.age.value_counts().plot(kind='bar')
display(rp.summary_cat(df_cleaned[['age']]))

# Average time in hospital 4.26 days 
display(df_cleaned.agg(
    {
    'num_medications':["min", "max", "mean","median", "skew"], 
    'num_lab_procedures':["min", "max", "mean","median", "skew"],
        'time_in_hospital':["min", "max", "mean","median", "skew"], 
        'num_lab_procedures':["min", "max", "mean","median", "skew"], 
                  'num_procedures':["min", "max", "mean","median", "skew"], 
        'num_medications':["min", "max", "mean","median", "skew"], 
                  'number_outpatient':["min", "max", "mean","median", "skew"], 
        'number_emergency':["min", "max", "mean","median", "skew"], 
        'number_inpatient':["min", "max", "mean","median", "skew"], 
        'number_diagnoses':["min", "max", "mean","median", "skew"]
    }
))

# average time in hospital slightly longer (4.78 days) for those readmitted less than 30 days vs those not readmitted (4.21 days)
display(df_cleaned.loc[:,['time_in_hospital', 'num_lab_procedures', 
                  'num_procedures', 'num_medications', 
                  'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']].groupby(df_cleaned['readmitted_recoded']).mean())

fig, ax = plt.subplots()

sum_cols = ['time_in_hospital', 'num_lab_procedures', 
                  'num_procedures', 'num_medications', 
                  'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
ax.boxplot(df_cleaned[sum_cols].values)
plt.xticks([1, 2, 3,4,5,6,7,8], sum_cols, rotation='vertical')
plt.show()

# df_cleaned.time_in_hospital.hist()
# plt.show()

###############################################################################################
## This code implements the Machine Learning model                                           ##
##                                                                                           ##
###############################################################################################                                   

y = encoded_df["readmitted_recoded_le"]
X = encoded_df.drop(columns=['readmitted_recoded_le'])

from sklearn.model_selection import train_test_split

# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X,
   y, random_state=1, stratify=y)

# # rebalance data 
# from imblearn.under_sampling import RandomUnderSampler
# from collections import Counter
# ros = RandomUnderSampler(random_state=1)
# X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
# Counter(y_resampled)

# Define the logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', max_iter=500, random_state=1)

print(f"\nEncoded dataframe is:\n{encoded_df.head}")
# Train the model
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)

d = {'Predicted': y_pred, 'Actual': y_test}
check_df = pd.DataFrame(data=d)
print(check_df.head)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Preprocess
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Define the logistic regression model
log_classifier = LogisticRegression(solver="lbfgs",max_iter=500)

# Train the model
log_classifier.fit(X_train_scaled,y_train)

# Evaluate the model
y_pred = log_classifier.predict(X_test_scaled)
print(f" Logistic regression model accuracy after scaling: {accuracy_score(y_test,y_pred):.3f}")

# Determine the shape of our training and testing sets.
print(X_train_scaled.shape)
print(X_test_scaled.shape)
print(y_train.shape)
print(y_test.shape)

# Get Feature Importance

# Create a random forest classifier.
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=128, random_state=78) 

# Fitting the model
model = model.fit(X_train_scaled, y_train)

# Calculate feature importance in the Random Forest model.
importances = model.feature_importances_

# We can sort the features by their importance.
feature_imp = sorted(zip(importances, X.columns), reverse=True)
for features in feature_imp:
    print(f"{features}")

X = encoded_df.drop(columns=['readmitted_recoded_le', 'pioglitazone_le', 'rosiglitazone_le', 'glimepiride_le', 'repaglinide_le',
                            'change_le', 'nateglinide_le', 'glyburide-metformin_le', 'diabetesmed_le', 'acarbose_le', 'chlorpropamide_le',
                            'tolbutamide_le', 'tolazamide_le', 'miglitol_le', 'glipizide-metformin_le', 'troglitazone_le', 'metformin-rosiglitazone_le',
                            'acetohexamide_le', 'metformin-pioglitazone_le', 'glimepiride-pioglitazone_le', 'examide_le', 'citoglipton_le'])

# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X,
   y, random_state=1, stratify=y)

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Define the logistic regression model
log_classifier = LogisticRegression(solver="lbfgs",max_iter=500)

# Train the model
log_classifier.fit(X_train_scaled,y_train)

# Evaluate the model
y_pred = log_classifier.predict(X_test_scaled)
print(f"\nLogistic regression model accuracy: {accuracy_score(y_test,y_pred):.3f}")

from sklearn.metrics import confusion_matrix, classification_report
matrix = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix: \n{matrix}")

report = classification_report(y_test, y_pred)
print(f"\nClassification Report: \n{report}")