import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Breast Cancer Wisconsin (Diagnostic) Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Support Vector Machine classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Function to predict diagnosis
def predict_diagnosis(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean):
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]])
    input_data_scaled = scaler.transform(input_data)
    prediction = svm_classifier.predict(input_data_scaled)
    return prediction[0]

# Streamlit UI
st.title('Breast Cancer Diagnosis')
st.write('This app predicts whether a breast cancer tumor is benign or malignant.')

radius_mean = st.slider('Radius Mean', min_value=0.0, max_value=40.0, value=10.0, step=0.1)
texture_mean = st.slider('Texture Mean', min_value=0.0, max_value=40.0, value=10.0, step=0.1)
perimeter_mean = st.slider('Perimeter Mean', min_value=0.0, max_value=300.0, value=100.0, step=1.0)
area_mean = st.slider('Area Mean', min_value=0.0, max_value=2500.0, value=500.0, step=1.0)
smoothness_mean = st.slider('Smoothness Mean', min_value=0.0, max_value=0.25, value=0.1, step=0.001)
compactness_mean = st.slider('Compactness Mean', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
concavity_mean = st.slider('Concavity Mean', min_value=0.0, max_value=1.0, value=0.3, step=0.01)
concave_points_mean = st.slider('Concave Points Mean', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
symmetry_mean = st.slider('Symmetry Mean', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
fractal_dimension_mean = st.slider('Fractal Dimension Mean', min_value=0.0, max_value=0.1, value=0.05, step=0.001)

if st.button('Predict Diagnosis'):
    prediction = predict_diagnosis(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean)
    if prediction == 0:
        st.write('Prediction: Benign')
    else:
        st.write('Prediction: Malignant')
