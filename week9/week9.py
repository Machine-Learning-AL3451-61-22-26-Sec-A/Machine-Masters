import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    st.title("Iris Flower Classification")
    st.write("This app predicts the species of iris flowers using the K-Nearest Neighbors algorithm.")
    
    # Load dataset
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=0)

    # Create and train the KNN classifier
    kn = KNeighborsClassifier(n_neighbors=1)
    kn.fit(X_train, y_train)

    # Make predictions and display results
    st.subheader("Prediction Results")
    for i, x in enumerate(X_test):
        x_new = np.array([x])
        prediction = kn.predict(x_new)
        st.write(f"Sample {i+1}:")
        st.write(f"   - Actual Target: {dataset.target_names[y_test[i]]}")
        st.write(f"   - Predicted Target: {dataset.target_names[prediction[0]]}")

    # Calculate and display the accuracy
    accuracy = kn.score(X_test, y_test)
    st.subheader("Model Accuracy")
    st.write(f"The accuracy of the model is: {accuracy:.2f}")

if __name__ == "__main__":
    main()
