import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import mlflow
import dagshub

import shap

from sklearn.ensemble._forest import RandomForestClassifier

@st.cache_data
def app_train_load(csv_file):
    app_train = pd.read_csv(csv_file)
    app_train['BIRTH_YEARS'] = app_train['DAYS_BIRTH'] / -365
    app_train = app_train.drop('DAYS_BIRTH', axis=1)

    app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

    app_train['DAYS_EMPLOYED'].replace({365243: np.nan})

    return app_train

@st.cache_data
def data_calcul(app_train):
    app_train_enc = app_train.copy()

    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in app_train_enc:
        if app_train_enc[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(app_train_enc[col].unique())) <= 2:
                # Train on the training data
                le.fit(app_train_enc[col])
                # Transform both training and testing data
                app_train_enc[col] = le.transform(app_train_enc[col])

                # Keep track of how many columns were label encoded
                le_count += 1

    app_train_enc = pd.get_dummies(app_train_enc)

    app_train_enc = app_train_enc.drop(columns = ['CODE_GENDER_XNA', 'NAME_INCOME_TYPE_Maternity leave', 'NAME_FAMILY_STATUS_Unknown'])

    data = app_train_enc.drop(columns = ['TARGET'])

    return data

@st.cache_data
def data_scaledMM_calcul(data):
    scaler_minMax = MinMaxScaler(feature_range = (0, 1))
    scaler_minMax.fit(data);

    data_scaledMM = scaler_minMax.transform(data)
    data_scaledMM = pd.DataFrame(data=data_scaledMM, columns = data.columns)
    return data_scaledMM

@st.cache_resource
def models_load():
    dagshub.auth.add_app_token('7ff59a8ec595a39c81790087b5fe632c13a71e8c')
    dagshub.init(repo_owner='jonathan.durand25', repo_name='OC_P7', mlflow=True)
    model = mlflow.pyfunc.load_model('runs:/4e1e5e9e3f3f48fda5eb52dbc836038c/model')
    model_learn = mlflow.sklearn.load_model('runs:/fa1ccfb2e0814e3d9261f7098c3d60c9/RF_sel_med')
    return model, model_learn

@st.cache_resource(hash_funcs={RandomForestClassifier: id})
def shap_values_calcul(model, row):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(row)
    return shap_vals