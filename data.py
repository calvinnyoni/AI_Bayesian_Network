from locale import normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read in data
data = pd.read_csv("data\heart.csv", sep=",")

#gender/sex data
gender = data.loc[:,"Sex"]
gender_counts = gender.value_counts()
norm_gender_counts = gender.value_counts(normalize=True)
print("--------------------------------------------------------------------")
print("Gender Counts")
print(gender_counts)
print("P(Gender)")
print(norm_gender_counts)
print("--------------------------------------------------------------------")

#age data
age = data.loc[:,"Age"]
age_counts = age.value_counts(bins=4, ascending=True)
norm_age_counts = age.value_counts(bins=4, ascending=True, normalize=True)
print("--------------------------------------------------------------------")
print("Age Counts")
print(age_counts)
print("P(Age)")
print(norm_age_counts)
print("Other stats")
print("Mean : {} Median : {} ".format(np.mean(age), np.median(age)))
print("Min : {} Max : {} ".format(np.amin(age), np.amax(age)))
print("Lower Q : {}  Upper Q : {} ".format(np.quantile(age, 0.25), np.quantile(age, 0.5)))
print("--------------------------------------------------------------------")

#excercise-induced angina data
angina = data.loc[:,"ExerciseAngina"]
angina_counts = angina.value_counts()
norm_angina_counts = angina.value_counts(normalize=True)
print("--------------------------------------------------------------------")
print("Angina Counts")
print(angina_counts)
print("P(ExerciseAngina)")
print(norm_angina_counts)
print("--------------------------------------------------------------------")

#cholesterol data
cholessterol = data.loc[:,"Cholesterol"]
print("--------------------------------------------------------------------")
print("Age vs Cholesterol Level Counts")
data['Age_Ranges'] = pd.cut(data["Age"], bins=np.arange(27.5, 77, 12.3, dtype=int)) #column for bins of ages for cross tabulation
data['Cholesterol_Ranges'] = pd.cut(data["Cholesterol"], [0, 200, 240, 603]) #column for bins of cholesterol for cross tabulation
age_chol_table = pd.crosstab(data['Cholesterol_Ranges'], data['Age_Ranges'])
print(age_chol_table)
print("P(Cholesterol|Age) Tabulation")
age_chol_ptable = pd.crosstab(data['Cholesterol_Ranges'], data['Age_Ranges'], normalize="columns")
print(age_chol_ptable)
print("Other stats")
print("Mean : {} Median : {} ".format(np.mean(cholessterol), np.median(cholessterol)))
print("Min : {} Max : {} ".format(np.amin(cholessterol), np.amax(cholessterol)))
print("Lower Q : {}  Upper Q : {} ".format(np.quantile(cholessterol, 0.25), np.quantile(cholessterol, 0.5)))
print("--------------------------------------------------------------------")

#max heart rate data
maxheartrate = data.loc[:,"MaxHR"]
print("--------------------------------------------------------------------")
print("Age vs Max Heart Rate Counts")
data['MaxHeartRate_Ranges'] = pd.cut(data["MaxHR"], [60, 140, 175, 220]) #column for bins of max heart rate for cross tabulation
age_maxhr_table = pd.crosstab(data['MaxHeartRate_Ranges'], data['Age_Ranges'])
print(age_maxhr_table)
print("P(MaxHeartRate|Age) Tabulation")
age_maxhr_ptable = pd.crosstab(data['MaxHeartRate_Ranges'], data['Age_Ranges'], normalize="columns")
print(age_maxhr_ptable)
print("Other stats")
print("Mean : {} Median : {} ".format(np.mean(maxheartrate), np.median(maxheartrate)))
print("Min : {} Max : {} ".format(np.amin(maxheartrate), np.amax(maxheartrate)))
print("Lower Q : {}  Upper Q : {} ".format(np.quantile(maxheartrate, 0.25), np.quantile(maxheartrate, 0.5)))
print("--------------------------------------------------------------------")

#resting ecg data
maxheartrate = data.loc[:,"RestingECG"]
print("--------------------------------------------------------------------")
print("Age vs Resting ECG Counts")
age_ecg_table = pd.crosstab(data['RestingECG'], data['Age_Ranges'])
print(age_ecg_table)
print("P(RestingECG|Age) Tabulation")
age_ecg_ptable = pd.crosstab(data['RestingECG'], data['Age_Ranges'], normalize="columns")
print(age_ecg_ptable)
print("--------------------------------------------------------------------")

#resting bp data
restingbp = data.loc[:,"RestingBP"]
print("--------------------------------------------------------------------")
print("Cholesterol vs Resting BP Counts")
data['RestingBP_Ranges'] = pd.cut(data["RestingBP"], [0, 120, 139, 200]) #column for bins of resting bp for cross tabulation
chol_restingbp_table = pd.crosstab(data['RestingBP_Ranges'], data['Cholesterol_Ranges'])
print(chol_restingbp_table)
print("P(RestingBP|Cholesterol) Tabulation")
chol_restingbp_ptable = pd.crosstab(data['RestingBP_Ranges'], data['Cholesterol_Ranges'], normalize="columns")
print(chol_restingbp_ptable)
print("Other stats")
print("Mean : {} Median : {} ".format(np.mean(restingbp), np.median(restingbp)))
print("Min : {} Max : {} ".format(np.amin(restingbp), np.amax(restingbp)))
print("Lower Q : {}  Upper Q : {} ".format(np.quantile(restingbp, 0.25), np.quantile(restingbp, 0.5)))
print("--------------------------------------------------------------------")

#chest pain type data
chestpain = data.loc[:,"ChestPainType"]
print("--------------------------------------------------------------------")
print("Chest Pain Type vs Excercise-Induced Angina Counts")
age_ecg_table = pd.crosstab(data['ChestPainType'], data['ExerciseAngina'])
print(age_ecg_table)
print("P(ChestPainType|ExerciseAngina) Tabulation")
age_ecg_ptable = pd.crosstab(data['ChestPainType'], data['ExerciseAngina'], normalize="columns")
print(age_ecg_ptable)
print("--------------------------------------------------------------------")

#heart disease data
heartdisease = data.loc[:,"HeartDisease"]
print("--------------------------------------------------------------------")
print("HeartDisease vs (RestingBP, MaxHR, Resting ECG) Counts")
heart_disease_table = pd.crosstab(data['HeartDisease'], [data['RestingBP_Ranges'], data['MaxHeartRate_Ranges'], data['RestingECG']])
print(heart_disease_table)
print("P(HeartDisease|RestingBP, MaxHR, RestingECG) Tabulation")
heart_disease_ptable = pd.crosstab(data['HeartDisease'], [data['RestingBP_Ranges'], data['MaxHeartRate_Ranges'], data['RestingECG']], normalize="columns")
print(heart_disease_ptable)
print("--------------------------------------------------------------------")