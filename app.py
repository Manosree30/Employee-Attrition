import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('xgb_attrition_model.pkl')

st.title("Employee Attrition Prediction App")
st.write("Predict whether an employee is likely to leave the company.")

# -----------------------------
# User Input
# -----------------------------
def user_input_features():
    data = {
        'Age': st.sidebar.slider('Age', 18, 60, 30),
        'DailyRate': st.sidebar.slider('DailyRate', 100, 2000, 1000),
        'DistanceFromHome': st.sidebar.slider('DistanceFromHome', 1, 30, 5),
        'Education': st.sidebar.selectbox('Education', [1, 2, 3, 4, 5]),
        'EmployeeCount': 1,
        'EmployeeNumber': st.sidebar.number_input('EmployeeNumber', min_value=1, max_value=1000, value=1),
        'EnvironmentSatisfaction': st.sidebar.slider('EnvironmentSatisfaction', 1, 4, 3),
        'Gender': st.sidebar.selectbox('Gender', ['Male', 'Female']),
        'HourlyRate': st.sidebar.slider('HourlyRate', 50, 200, 100),
        'JobInvolvement': st.sidebar.slider('JobInvolvement', 1, 4, 3),
        'JobLevel': st.sidebar.slider('JobLevel', 1, 5, 1),
        'JobSatisfaction': st.sidebar.slider('JobSatisfaction', 1, 4, 3),
        'MonthlyIncome': st.sidebar.slider('MonthlyIncome', 1000, 20000, 5000),
        'MonthlyRate': st.sidebar.slider('MonthlyRate', 1000, 20000, 5000),
        'NumCompaniesWorked': st.sidebar.slider('NumCompaniesWorked', 0, 10, 1),
        'Over18': 1,
        'OverTime': st.sidebar.selectbox('OverTime', ['Yes', 'No']),
        'PercentSalaryHike': st.sidebar.slider('PercentSalaryHike', 0, 50, 10),
        'PerformanceRating': st.sidebar.slider('PerformanceRating', 1, 4, 3),
        'RelationshipSatisfaction': st.sidebar.slider('RelationshipSatisfaction', 1, 4, 3),
        'StandardHours': 80,
        'StockOptionLevel': st.sidebar.slider('StockOptionLevel', 0, 3, 0),
        'TotalWorkingYears': st.sidebar.slider('TotalWorkingYears', 0, 40, 5),
        'TrainingTimesLastYear': st.sidebar.slider('TrainingTimesLastYear', 0, 10, 2),
        'WorkLifeBalance': st.sidebar.slider('WorkLifeBalance', 1, 4, 3),
        'YearsAtCompany': st.sidebar.slider('YearsAtCompany', 0, 40, 3),
        'YearsInCurrentRole': st.sidebar.slider('YearsInCurrentRole', 0, 20, 2),
        'YearsSinceLastPromotion': st.sidebar.slider('YearsSinceLastPromotion', 0, 15, 1),
        'YearsWithCurrManager': st.sidebar.slider('YearsWithCurrManager', 0, 20, 2),
        # Categorical one-hot encoded columns
        'BusinessTravel_Travel_Frequently': 0,
        'BusinessTravel_Travel_Rarely': 0,
        'Department_Research & Development': 0,
        'Department_Sales': 0,
        'EducationField_Life Sciences': 0,
        'EducationField_Marketing': 0,
        'EducationField_Medical': 0,
        'EducationField_Other': 0,
        'EducationField_Technical Degree': 0,
        'JobRole_Human Resources': 0,
        'JobRole_Laboratory Technician': 0,
        'JobRole_Manager': 0,
        'JobRole_Manufacturing Director': 0,
        'JobRole_Research Director': 0,
        'JobRole_Research Scientist': 0,
        'JobRole_Sales Executive': 0,
        'JobRole_Sales Representative': 0,
        'MaritalStatus_Married': 0,
        'MaritalStatus_Single': 0
    }

    # Convert binary categories
    data['Gender'] = 1 if data['Gender'] == 'Male' else 0
    data['OverTime'] = 1 if data['OverTime'] == 'Yes' else 0

    # Let user select one-hot category for BusinessTravel
    bt = st.sidebar.selectbox('BusinessTravel', ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
    if bt == 'Travel_Rarely':
        data['BusinessTravel_Travel_Rarely'] = 1
    elif bt == 'Travel_Frequently':
        data['BusinessTravel_Travel_Frequently'] = 1

    # Department
    dept = st.sidebar.selectbox('Department', ['Sales', 'Research & Development'])
    if dept == 'Sales':
        data['Department_Sales'] = 1
    else:
        data['Department_Research & Development'] = 1

    # EducationField
    edu = st.sidebar.selectbox('EducationField', ['Life Sciences', 'Medical', 'Marketing', 'Other', 'Technical Degree'])
    key = f'EducationField_{edu}'
    if key in data:
        data[key] = 1

    # JobRole
    role = st.sidebar.selectbox('JobRole', [
        'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 
        'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'])
    key = f'JobRole_{role}'
    if key in data:
        data[key] = 1

    # MaritalStatus
    ms = st.sidebar.selectbox('MaritalStatus', ['Married', 'Single'])
    key = f'MaritalStatus_{ms}'
    if key in data:
        data[key] = 1

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# -----------------------------
# Ensure order of columns matches training
# -----------------------------
train_cols = ['Age', 'Attrition', 'DailyRate', 'DistanceFromHome', 'Education',
       'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender',
       'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18',
       'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently',
       'BusinessTravel_Travel_Rarely', 'Department_Research & Development',
       'Department_Sales', 'EducationField_Life Sciences',
       'EducationField_Marketing', 'EducationField_Medical',
       'EducationField_Other', 'EducationField_Technical Degree',
       'JobRole_Human Resources', 'JobRole_Laboratory Technician',
       'JobRole_Manager', 'JobRole_Manufacturing Director',
       'JobRole_Research Director', 'JobRole_Research Scientist',
       'JobRole_Sales Executive', 'JobRole_Sales Representative',
       'MaritalStatus_Married', 'MaritalStatus_Single']

# Add any missing columns with 0
for col in train_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder
input_df = input_df[train_cols]

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

st.subheader("Prediction")
st.write("Employee will **Leave**" if prediction==1 else "**Stay**")

st.subheader("Probability of Leaving")
st.write(f"{prediction_proba:.2f}")
