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
cur.execute("DROP TABLE IF EXISTS medicines_info");
cur.execute("DROP TABLE IF EXISTS admission_info");
cur.execute("DROP TABLE IF EXISTS diagnosis_info");
cur.execute("DROP TABLE IF EXISTS patient_info");
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
cur.execute(cmd)

###############################################################################################
## This code implements the Data Cleaning Logic                                              ##
##                                                                                           ##
###############################################################################################

# Create the connection to the PostgreSQL database
db_string = f"postgres://{DB_USER}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_string)
conn.commit()

# Read the raw data from the postgres into dataframe
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

# In order to keep observations independent, only one the first encounter is included
# Dedupe based on first encounter
df_deduped = df_copy.drop_duplicates(subset=['patient_nbr'], keep='first')

# Remove encounters that resulted in either discharge to 
# a hospice or patient death to avoid biasing analysis 
discharge_disposition_excluded=[11, 13, 14, 19, 20, 23]

df_cleaned = df_deduped[~df_deduped.discharge_disposition_id.isin(discharge_disposition_excluded)]

df_cleaned['gender'].value_counts()

df_cleaned.drop(df_cleaned[(df_cleaned.loc[:,'gender'] == "Unknown/Invalid")].index, inplace = True)
df_cleaned['gender'].value_counts()

# recode readmitted to be binary 
def recode_readmit(x):
    if x == '<30':
        return 1
    else:
        return 0

df_cleaned['readmitted_recoded'] = df_cleaned['readmitted'].apply(recode_readmit)
# df_cleaned['readmitted_recoded'] = df_cleaned[df_cleaned.loc[:,'readmitted']].apply(recode_readmit)
# df_cleaned['readmitted_recoded'] = df_cleaned.loc[:,['readmitted']].replace({'>30': 'NO'})

# recode medical_specialty to add missing 
df_cleaned['medical_specialty_recoded'] = df_cleaned.loc[:,['medical_specialty']].replace("?",'missing', inplace=False)

# readmitted recoded is imbalanced with 91% of cases 
# not having a 30 day readmission
print(collections.Counter(df_cleaned['readmitted_recoded']))

df_cleaned['medical_specialty_recoded'].value_counts()

df_cleaned['medical_specialty_recoded'] = df_cleaned.loc[:,['medical_specialty_recoded']].replace({'Family/GeneralPractice': 'InternalMedicine'})

def values_to_other(col_name, value_unchanged):
    v=[]
    for value in col_name:
        if (value != value_unchanged):
            v.append("Other")
        else:
            v.append(value)
    return v
    

df_cleaned['medical_specialty_recoded'] = values_to_other(df_cleaned['medical_specialty_recoded'].values,"InternalMedicine")    

df_cleaned['medical_specialty_recoded'].value_counts()

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

# replace 'age' values with cleaned values
df_cleaned['age'] = parse_age_range(df_cleaned['age'].values)

df_cleaned = df_cleaned.drop(columns=['readmitted', 'medical_specialty'])

print(df_cleaned.columns)

print(df_cleaned.dtypes)


# # DB code goes here
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


###############################################################################################
## This code implements the Exploratory Data Analysis                                        ##
##                                                                                           ##
############################################################################################### 

# Read the raw data from the postgres into dataframe
df_cleaned = pd.read_sql_table("diabetes_clean_data", engine)
print(df_cleaned.head)

# 69,710 records in the dataset 
print(df_cleaned.describe())

# 53.25% of admissions came from Trauma Center followed by ED
print(rp.summary_cat(df_cleaned[['admission_source_id']]))

# ~75% of cases were caucasian followed by 18% for African American 
print(rp.summary_cat(df_cleaned[['race']]))

# 53% cases were female, 47% male 
print(rp.summary_cat(df_cleaned[['gender']]))
df_cleaned.gender.value_counts().plot(kind='bar', title='gender')
plt.show()

#~81% of cases above age 50
df_cleaned.age.value_counts().plot(kind='bar')
print(rp.summary_cat(df_cleaned[['age']]))

# Average time in hospital 4.26 days 
df_cleaned.agg(
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
)

# average time in hospital slightly longer (4.78 days) for those readmitted less than 30 days vs those not readmitted (4.21 days)
df_cleaned.loc[:,['time_in_hospital', 'num_lab_procedures', 
                  'num_procedures', 'num_medications', 
                  'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']].groupby(df_cleaned['readmitted_recoded']).mean()


fig, ax = plt.subplots()

sum_cols = ['time_in_hospital', 'num_lab_procedures', 
                  'num_procedures', 'num_medications', 
                  'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
ax.boxplot(df_cleaned[sum_cols].values)
plt.xticks([1, 2, 3,4,5,6,7,8], sum_cols, rotation='vertical')
plt.show()

print(df_cleaned.head(5))

df_cleaned.time_in_hospital.hist()


###############################################################################################
## This code implements the Data Pre-Processing for Machine Learning model                   ##
##                                                                                           ##
###############################################################################################  

encoded_df = df_cleaned.copy()

print(df_cleaned.columns)

# Show missing in dataset 
for column in df_cleaned.columns:
    missing_count = df_cleaned[column].isnull().sum()
    if missing_count>0:
        print(column,":",df_cleaned)

df_cleaned.drop(df_cleaned[df_cleaned['num_lab_procedures'] > 95].index, inplace = True) 
df_cleaned.drop(df_cleaned[df_cleaned['num_medications'] > 32].index, inplace = True) 
df_cleaned.drop(df_cleaned[df_cleaned['number_diagnoses'] > 13].index, inplace = True)
df_cleaned.drop(df_cleaned[df_cleaned['num_procedures'] > 5].index, inplace = True)

# Create a list of columns to encode for each variable if variable type is object
columns_to_encode = [column for column in df_cleaned.columns if df_cleaned[column].dtypes == 'O']

print(columns_to_encode)


# exclude diag 1, 2, and 3 from recode list 
for i in range(3):
    to_remove = 'diag_' + str(i+1)
    columns_to_encode.remove(to_remove)
print(columns_to_encode)
print("\n", len(columns_to_encode), "columns")

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

print(encoded_df.columns)

cols_to_drop = ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'admission_type_id','max_glu_serum', 'a1cresult', 'metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
       'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
       'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesmed', 'readmitted_recoded',
       'medical_specialty_recoded', 'diag_1', 'diag_2', 'diag_3']
encoded_df = encoded_df.drop(columns=cols_to_drop)


###############################################################################################
## This code implements the Machine Learning model                                           ##
##                                                                                           ##
###############################################################################################  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
# from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from imblearn.metrics import classification_report_imbalanced
from sklearn.preprocessing import StandardScaler
from collections import Counter

print(encoded_df.head(5))

from sklearn.model_selection import train_test_split
y = encoded_df["readmitted_recoded_le"]
X = encoded_df.drop(columns=['readmitted_recoded_le'])
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1, stratify=y)

def list_feature_imp(X_train_input, y_train_input):
    x_values = []
    y_values = []
    # Create a random forest classifier.
    rf_model = RandomForestClassifier(n_estimators=128, random_state=78) 
    # Fitting the model
    rf_model = rf_model.fit(X_train_input, y_train_input)
    # Calculate feature importance in the Random Forest model.
    importances = rf_model.feature_importances_
    feature_imp = sorted(zip(importances, X.columns), reverse=True)
    for features in feature_imp:
        print(f"{features}")
        x_values.append(features[0])
        y_values.append(features[1])
#     Plot the feature importances of the forest
    fig, ax = plt.subplots(figsize=(10,15))
    ax.barh(y_values, x_values, align='center')
#     ax.text(0.15, y_values, x_values, fontsize=12)
    ax.invert_yaxis()

list_feature_imp(X_train, y_train)


X = X.drop(columns=['acetohexamide_le', 
                             'tolbutamide_le', 
                             'miglitol_le', 
                             'troglitazone_le',
                             'metformin-rosiglitazone_le', 
                             'metformin-pioglitazone_le', 
                             'glipizide-metformin_le', 
                             'glimepiride-pioglitazone_le', 
                             'examide_le', 
                             'citoglipton_le',
                             'glyburide-metformin_le',
                             'acarbose_le',
                             'chlorpropamide_le',
                             'tolazamide_le',
                            'number_emergency',
                            'number_outpatient',
                             'number_inpatient',
                            'nateglinide_le',
                            'repaglinide_le',
                            'diabetesmed_le',
                            'max_glu_serum_le'
                            ])


# Split training/test datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1, stratify=y)


# ros = RandomUnderSampler(random_state=1)
# X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
# Counter(y_resampled)

from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
Counter(y_resampled)

# Creating a StandardScaler instance.
scaler = StandardScaler()
# Fitting the Standard Scaler with the training data.
X_scaler = scaler.fit(X_resampled)

# Scaling the data.
X_train_scaled = X_scaler.transform(X_resampled)
X_test_scaled = X_scaler.transform(X_test)

from yellowbrick.classifier import ClassificationReport
import matplotlib.gridspec as gridspec
def apply_ml_model(X_train_input, y_train_input, X_test_input, y_test_input):
    models = ['LREG','RFC','Tree','Balanced RFC']
    scores = []
    # Specify the target classes
    classes = ["No re-admission","Re-admission in < 30 days"]
    for model in models:
        if model == 'LREG':
            model_select = LogisticRegression(solver='lbfgs', max_iter=500, random_state=78)
        elif model == 'RFC':
            model_select = RandomForestClassifier(n_estimators= 128, random_state=78)
        elif model == 'Tree':
            model_select = tree.DecisionTreeClassifier(random_state=78)
        elif model == 'Balanced RFC':
            model_select = BalancedRandomForestClassifier(n_estimators=128, random_state=78)
        model_select.fit(X_train_input, y_train_input)
        y_pred = model_select.predict(X_test_input)
        # Create a DataFrame from the confusion matrix.
        cm = confusion_matrix(y_test_input, y_pred)
        # Calculating the accuracy score.
        acc_score = balanced_accuracy_score(y_test, y_pred)
        scores.append(acc_score)
        print(f"Model: {model}")
        # Displaying results
        print("Confusion Matrix")
        cm_df = pd.DataFrame(
        cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        print(cm_df)
        print(f"Accuracy Score : {acc_score}\n")
        print("Classification Report")
        print(classification_report_imbalanced(y_test_input, y_pred))
        visualizer = ClassificationReport(model_select, classes=classes, support=True)
        visualizer.fit(X_train_input, y_train_input)        # Fit the visualizer and the model
        visualizer.score(X_test_input, y_test_input)        # Evaluate the model on the test data
        visualizer.show()                       # Finalize and show the figure


apply_ml_model(X_train_scaled, y_resampled, X_test_scaled, y_test)


