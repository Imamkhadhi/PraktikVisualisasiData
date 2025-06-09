import streamlit as st
import joblib
import numpy as np

# Load models once
model_dt = joblib.load("modelJb_DecisionTree.joblib")
model_knn = joblib.load("modelJb_KNN.joblib")
model_svm = joblib.load("modelJb_SVM.joblib")
model_nn = joblib.load("modelJb_NN.joblib")


def predict_and_show(model_name, model, input_data):
    pred = model.predict(input_data)
    label_map = {
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica"
    }
    label = label_map.get(pred[0], "Unknown")
    st.subheader(f"{model_name} Prediction: {pred[0]} → {label}")


def show_single():
    st.title("Single Prediction")

    # Input fields
    a = st.number_input("Sepal length in cm", min_value=0.0, step=0.1)
    b = st.number_input("Sepal width in cm", min_value=0.0, step=0.1)
    c = st.number_input("Petal length in cm", min_value=0.0, step=0.1)
    d = st.number_input("Petal width in cm", min_value=0.0, step=0.1)

    # Model checkboxes
    use_knn = st.checkbox("Use KNN")
    use_svm = st.checkbox("Use SVM")
    use_nn = st.checkbox("Use Neural Network")
    use_dt = st.checkbox("Use Decision Tree")

    if st.button("Predict"):
        input_data = np.array([[a, b, c, d]])

        if use_knn:
            predict_and_show("K-Nearest Neighbors", model_knn, input_data)
        if use_svm:
            predict_and_show("Support Vector Machine", model_svm, input_data)
        if use_nn:
            predict_and_show("Neural Network", model_nn, input_data)
        if use_dt:
            predict_and_show("Decision Tree", model_dt, input_data)

        if not any([use_knn, use_svm, use_nn, use_dt]):
            st.warning("Please select at least one model.")