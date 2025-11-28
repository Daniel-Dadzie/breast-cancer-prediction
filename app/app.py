import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level to project root

# Load the model, scaler, and metadata
try:
    model_path = os.path.join(project_root, 'models', 'breast_cancer_model.pkl')
    scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
    metadata_path = os.path.join(project_root, 'models', 'model_metadata.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model files not found. Please train the model first using the notebook.")
    st.stop()

# Sidebar
st.sidebar.title("About the Project")
st.sidebar.write("""
This app uses machine learning to predict whether a breast mass is benign or malignant based on cellular features.
""")
st.sidebar.write(f"**Model Used:** {metadata['model_name']}")
st.sidebar.write(f"**Model Accuracy:** {metadata['accuracy']:.2f}")
st.sidebar.write("**Dataset:** Breast Cancer Wisconsin (Diagnostic)")
st.sidebar.markdown("[Dataset Source](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)")

# How to Use
with st.sidebar.expander("How to Use"):
    st.write("""
    1. Enter the values for the 30 cellular features below.
    2. Click the 'Predict' button.
    3. View the prediction result and probabilities.
    Note: Values should be based on medical measurements.
    """)

# Title
st.title("Breast Cancer Prediction System")

st.write("""
Enter the cellular features to predict if the breast mass is benign or malignant.
""")

# Input features
st.header("Input Features")

# Use features from metadata to ensure consistency
feature_names = metadata['features']
# Use original dataset means as default values instead of scaled means
original_means = [
    14.127, 19.290, 91.969, 654.889, 0.096, 0.104, 0.089, 0.049, 0.181, 0.063,
    0.405, 1.217, 2.866, 40.337, 0.007, 0.025, 0.032, 0.012, 0.021, 0.004,
    16.269, 25.677, 107.261, 880.583, 0.132, 0.254, 0.272, 0.115, 0.290, 0.084
]

# Group features into tabs for better organization
tab1, tab2, tab3 = st.tabs(["Mean Features", "Error Features", "Worst Features"])

inputs = {}

with tab1:
    st.subheader("Mean Features (1-10)")
    col1, col2 = st.columns(2)
    for i in range(10):
        if i < 5:
            with col1:
                inputs[feature_names[i]] = st.number_input(f"{feature_names[i]}", value=original_means[i], step=0.01, key=feature_names[i])
        else:
            with col2:
                inputs[feature_names[i]] = st.number_input(f"{feature_names[i]}", value=original_means[i], step=0.01, key=feature_names[i])

with tab2:
    st.subheader("Error Features (11-20)")
    col1, col2 = st.columns(2)
    for i in range(10, 20):
        if i < 15:
            with col1:
                inputs[feature_names[i]] = st.number_input(f"{feature_names[i]}", value=original_means[i], step=0.01, key=feature_names[i])
        else:
            with col2:
                inputs[feature_names[i]] = st.number_input(f"{feature_names[i]}", value=original_means[i], step=0.01, key=feature_names[i])

with tab3:
    st.subheader("Worst Features (21-30)")
    col1, col2 = st.columns(2)
    for i in range(20, 30):
        if i < 25:
            with col1:
                inputs[feature_names[i]] = st.number_input(f"{feature_names[i]}", value=original_means[i], step=0.01, key=feature_names[i])
        else:
            with col2:
                inputs[feature_names[i]] = st.number_input(f"{feature_names[i]}", value=original_means[i], step=0.01, key=feature_names[i])

# Prediction
if st.button("Predict"):
    # Validate inputs
    if len(inputs) != len(feature_names):
        st.error(f"Expected {len(feature_names)} features, got {len(inputs)}")
        st.stop()

    input_values = list(inputs.values())

    # Check for empty or unrealistic inputs
    if all(v == 0.0 for v in input_values):
        st.warning("⚠️ All inputs are zero. Please enter realistic medical measurement values for accurate predictions.")
        st.stop()

    # Check for negative values (measurements shouldn't be negative)
    if any(v < 0 for v in input_values):
        st.error("❌ Negative values are not allowed for medical measurements.")
        st.stop()

    # Create DataFrame with feature names for proper scaling
    input_df = pd.DataFrame([input_values], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("🟢 Benign — Low Risk")
    else:
        st.error("🔴 Malignant — High Risk")

    st.write(f"**Probability of Benign:** {prediction_proba[0][1]:.2f}")
    st.write(f"**Probability of Malignant:** {prediction_proba[0][0]:.2f}")