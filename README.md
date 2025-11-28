# Breast Cancer Prediction Project

## Overview

This machine learning project predicts breast cancer diagnosis using the Wisconsin Breast Cancer Dataset. The project includes data preprocessing, model training, evaluation, and a Streamlit web app for predictions.

## Project Structure

```
breast_cancer_project/
├── notebooks/
│   └── breast_cancer_notebook.ipynb       # Your full ML notebook (run with Colab)
├── models/
│   ├── breast_cancer_model.pkl            # Saved best model
│   ├── scaler.pkl                         # Saved StandardScaler
│   └── model_metadata.pkl                 # Optional metadata file
├── app/
│   └── app.py                             # Streamlit app
├── reports/
│   └── (project report - to be created)   # Optional project report
├── data/
│   ├── data.csv                 # Dataset from Kaggle
│   └── (optional) dumps, EDA images, etc.
└── README.md                              # Overview + instructions
```

## Setup Instructions

1. **Clone or navigate to the project directory**

2. **Download the dataset:**
   - Download `data.csv` from Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
   - Place the file in the `data/` directory

3. **Install required dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the Jupyter notebook to train models and generate saved files:**
   ```
   jupyter notebook notebooks/breast_cancer_notebook.ipynb
   ```
    Execute all cells in the notebook to train the models and save them to the `models/` directory.

5. **Run the Streamlit app:**
   ```
   streamlit run app/app.py
   ```

## Usage

- The notebook contains complete EDA, preprocessing, model training (Logistic Regression, Random Forest, SVM), evaluation, and model saving.
- The Streamlit app provides an interactive interface for breast cancer prediction with input fields for cellular features.
- The project demonstrates ML model training, evaluation, and deployment for medical prediction.

## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset from Kaggle (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), containing 569 samples with 30 features describing cell nucleus characteristics.

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter
- streamlit
- joblib