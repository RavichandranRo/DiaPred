# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:04:40 2023

@author: chell
"""


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

diabetes_df = pd.read_csv(r'C:\Users\chell\Desktop\Ravi Project\diabetes.csv')
X = diabetes_df.drop(columns=['Outcome'])
y = diabetes_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

def app():st.title('Diabetes Prediction')
st.write('Please enter the following information for prediction:')
pregnancies = st.slider('Number of Pregnancies', 0, 17, 1)
glucose = st.slider('Glucose Level (mg/dL)', 0, 200, 100)
blood_pressure = st.slider('Blood Pressure (mm Hg)', 0, 122, 70)
skin_thickness = st.slider('Skin Thickness (mm)', 0, 99, 20)
insulin = st.slider('Insulin Level (mu U/ml)', 0, 846, 79)
bmi = st.slider('BMI (Body Mass Index)', 0.0, 67.1, 25.0)
dpf = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
age = st.slider('Age (years)', 21, 81, 30)

predict_button = st.button('Predict')
result = st.empty()

# Define the prediction function
def predict():
    # Prepare the input data for prediction
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                               columns=X.columns)
    # Make prediction using the trained model
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]  # Probability of positive class (diabetic)
    
    # Display the prediction result
    if prediction[0] == 0:
        result.markdown('**Result:** You are not diabetic.', unsafe_allow_html=True)
    else:
        result.markdown(f'**Result:** You are diabetic with a probability of {prediction_proba[0]:.2f}.', 
                        unsafe_allow_html=True)

# Handle button click event
if predict_button:
    predict()

if __name__ == '__main__':
    app()