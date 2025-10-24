
# Employee Attrition Prediction Using XGBoost
## Overview
This project aims to **predict employee attrition** (whether an employee will leave the company) using machine learning techniques. The model uses **XGBoost**, along with preprocessing steps such as outlier handling, skewness reduction, and categorical encoding, to accurately classify employee attrition.  

The project is based on the **IBM HR Analytics dataset** from Kaggle: [Dataset Link](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## Streamlit app
 App link-Employee Attrition(https://manosree30-employee-attrition-app-rrokww.streamlit.app/)

## Features
The dataset contains employee information, including:

- Personal information: Age, Gender, Marital Status, Education, etc.
- Work environment: Department, Job Role, Business Travel, OverTime, Distance from Home
- Compensation: Daily Rate, Monthly Income, Stock Option Level
- Performance and satisfaction: Years at Company, Total Working Years, Work-Life Balance, Relationship Satisfaction

---
## Preprocessing Steps
1. **Mapping binary columns**: Yes/No → 1/0, Male/Female → 1/0  
2. **Outlier capping**: Values beyond the 1st and 99th percentile capped  
3. **Skewness reduction**: Numeric features transformed using **Yeo-Johnson**  
4. **Encoding categorical variables**: One-hot encoding for multi-category features  
5. **Scaling**: StandardScaler applied to numeric features  

---

## Model
- **Algorithm**: XGBoost Classifier  
- **Pipeline**: Preprocessing + XGBoost  
- **Hyperparameter Tuning**:
  - `learning_rate`: 0.05, 0.1  
  - `n_estimators`: 100, 300  
  - `max_depth`: 3, 5  
  - `subsample`: 0.8, 1.0  
  - `colsample_bytree`: 0.8, 1.0  
- **Cross-validation**: 5-fold stratified  
---

## Evaluation Metrics
The model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
Example result:
Accuracy: 0.88
Precision: 0.74
Recall: 0.67
F1-score: 0.70
ROC-AUC: 0.815

## Visualization
- Boxplots of numeric features after outlier capping and skewness reduction
- Histograms to visualize distributions
- ROC curve for model performance
- Feature importance bar plot

## Conclusion

This project provides a robust pipeline to predict employee attrition, helping HR teams identify high-risk employees and take preventive actions. The XGBoost model shows good predictive performance with a ROC-AUC of 0.815.
