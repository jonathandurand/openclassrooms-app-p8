import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import streamviz

import mlflow

import dagshub

def init():
    dagshub.auth.add_app_token('7ff59a8ec595a39c81790087b5fe632c13a71e8c')
    dagshub.init(repo_owner='jonathan.durand25', repo_name='OC_P7', mlflow=True)

    app_train = pd.read_csv('application_train.csv')
    #app_train = pd.read_csv('https://drive.google.com/file/d/16RR6zIzq2JKPMXHsT8jirXaW62jGc8W6/view?usp=drive_link')
    print("Lecture données OK")
    print(app_train.shape)

    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in app_train:
        if app_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(app_train[col].unique())) <= 2:
                # Train on the training data
                le.fit(app_train[col])
                # Transform both training and testing data
                app_train[col] = le.transform(app_train[col])

                # Keep track of how many columns were label encoded
                le_count += 1

    app_train = pd.get_dummies(app_train)

    app_train = app_train.drop(columns = ['CODE_GENDER_XNA', 'NAME_INCOME_TYPE_Maternity leave', 'NAME_FAMILY_STATUS_Unknown'])

    app_train['BIRTH_YEARS'] = app_train['DAYS_BIRTH'] / -365
    app_train = app_train.drop('DAYS_BIRTH', axis=1)

    # Create an anomalous flag column
    app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan})
    st.session_state['app_train'] = app_train

    print('app_train encoded')
    print(app_train.shape)

    data = app_train.drop(columns = ['TARGET'])

    scaler_minMax = MinMaxScaler(feature_range = (0, 1))
    scaler_minMax.fit(data);

    data_scaledMM = scaler_minMax.transform(data)
    st.session_state['data_scaledMM'] = data_scaledMM

    print('data scaled')
    print(data_scaledMM.shape)

    st.session_state['model'] = mlflow.sklearn.load_model('runs:/4c4532ba1c904fcc98b806296cb62f6a/RF_full')
    print('model loaded')

def main():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

    st.session_state['client'] = st.selectbox(
        "How would you like to be contacted?",
        st.session_state['app_train']['SK_ID_CURR'].values,
        index=None,
        placeholder="Select client"
    )

    if st.session_state['client']!=None:
        print(st.session_state['client'])
        st.session_state['index'] = np.where(st.session_state['app_train']['SK_ID_CURR']==st.session_state['client'])
        streamviz.gauge(
            0.5
            )
        #st.session_state['model'].predict_proba(st.session_state['data_scaledMM'][st.session_state['index']].reshape(1,239))[:,1]

if __name__ == "__main__":
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        init()
        st.session_state.initialized = True
    
    main()