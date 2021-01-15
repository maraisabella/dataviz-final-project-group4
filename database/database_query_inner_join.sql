-- Deliverable 2
-- Drop all the existing tables
DROP TABLE IF EXISTS medicines_info;
DROP TABLE IF EXISTS admission_info;
DROP TABLE IF EXISTS diagnosis_info;
DROP TABLE IF EXISTS patient_info;
DROP TABLE IF EXISTS diabetes_clean_data;
DROP TABLE IF EXISTS diabetes_raw_data;

-- Create diabetes_raw_data to store the raw data from the csv file
CREATE TABLE diabetes_raw_data (
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
    UNIQUE (encounter_id));
	
SELECT * FROM diabetes_raw_data;

COPY diabetes_raw_data
    FROM '/Users/anusuyapoonja/Bootcamp_Analysis/Modules/Project-Group4/dataviz-final-project-group4/database/diabetic_data_initial.csv'
    CSV HEADER DELIMITER ',';

SELECT * FROM diabetes_raw_data;


-- Create diabetes_clean_data to store the data after cleaning it.
-- The data into the diabetes_clean_data will be imported from the python script
CREATE TABLE diabetes_clean_data (
    encounter_id int NOT NULL,
    patient_nbr	int NOT NULL,
    race varchar(20),
    gender varchar(20) NOT NULL,
    age varchar(10) NOT NULL,
    admission_type_id int NOT NULL,
    discharge_disposition_id int NOT NULL,
    admission_source_id int NOT NULL,
    time_in_hospital int NOT NULL,
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
	readmitted_recoded  varchar(10),
	medical_specialty_recoded varchar(40),
    PRIMARY KEY (encounter_id),
    UNIQUE (encounter_id));
	
SELECT * FROM diabetes_clean_data;

CREATE TABLE Patient (
	encounter_id int NOT NULL,
	patient_nbr	int NOT NULL,
	race varchar(20),
	gender varchar(20) NOT NULL,
	age varchar(10) NOT NULL,
	FOREIGN KEY (encounter_id) REFERENCES diabetes_clean_data (encounter_id),
	PRIMARY KEY (encounter_id)
);

INSERT INTO Patient (encounter_id, patient_nbr, race, gender, age)
SELECT DISTINCT encounter_id, patient_nbr, race, gender, age
FROM diabetes_clean_data
ON CONFLICT (encounter_id) DO NOTHING;

SELECT * FROM patient;

CREATE TABLE Admission (
	encounter_id int NOT NULL,
	patient_nbr	int NOT NULL,
	admission_type_id int NOT NULL,
	discharge_disposition_id int NOT NULL,
	admission_source_id int NOT NULL,
	time_in_hospital int NOT NULL,
	medical_specialty_recoded varchar(40),
	FOREIGN KEY (encounter_id) REFERENCES diabetes_clean_data (encounter_id),
	PRIMARY KEY (encounter_id)	
);

INSERT INTO Admission (encounter_id, patient_nbr, admission_type_id, discharge_disposition_id,
					   admission_source_id, time_in_hospital)
SELECT DISTINCT encounter_id, patient_nbr, admission_type_id, discharge_disposition_id,
					   admission_source_id, time_in_hospital
FROM diabetes_clean_data
ON CONFLICT (encounter_id) DO NOTHING;

SELECT * FROM Admission;

CREATE TABLE Diagnosis (
	encounter_id int NOT NULL,
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
	change  varchar(10),
	diabetesMed  varchar(10),
	readmitted_recoded  varchar(10),
	FOREIGN KEY (encounter_id) REFERENCES diabetes_clean_data (encounter_id),
	PRIMARY KEY (encounter_id)
);

INSERT INTO Diagnosis (encounter_id, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, 
					  number_inpatient, diag_1, diag_2, diag_3, number_diagnoses, max_glu_serum, A1Cresult, 
					  change, diabetesMed, readmitted_recoded)
SELECT DISTINCT encounter_id, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency,
					number_inpatient, diag_1, diag_2, diag_3, number_diagnoses, max_glu_serum, A1Cresult, 
					change, diabetesMed, readmitted_recoded		
FROM diabetes_clean_data
ON CONFLICT (encounter_id) DO NOTHING;

SELECT * FROM Diagnosis;

CREATE TABLE Medicines (
	encounter_id int NOT NULL,
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
	FOREIGN KEY (encounter_id) REFERENCES diabetes_clean_data (encounter_id),
	PRIMARY KEY (encounter_id)
);

INSERT INTO Medicines (encounter_id, metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide,
					   glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone,
					   tolazamide, examide, citoglipton, insulin, "glyburide-metformin", "glipizide-metformin",
					   "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone")
SELECT DISTINCT encounter_id, metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide,
				glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone,
				tolazamide, examide, citoglipton, insulin, "glyburide-metformin", "glipizide-metformin",
				"glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
FROM diabetes_clean_data
ON CONFLICT (encounter_id) DO NOTHING;

SELECT * FROM Medicines;

-- Joining patient and diagnosis tables
SELECT patient.patient_nbr,
    patient.gender,
	patient.age,
	diagnosis.number_diagnoses,
	diagnosis.diabetesmed,
	diagnosis.readmitted_recoded
FROM patient
INNER JOIN diagnosis
ON patient.encounter_id = diagnosis.encounter_id;