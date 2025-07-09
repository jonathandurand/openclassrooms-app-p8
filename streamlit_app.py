import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import plotly.graph_objects as go

import mlflow

import dagshub

def init():
    dagshub.auth.add_app_token('7ff59a8ec595a39c81790087b5fe632c13a71e8c')
    dagshub.init(repo_owner='jonathan.durand25', repo_name='OC_P7', mlflow=True)

    app_train = pd.read_csv('datas/app_train_1.csv')
    #app_train_2 = pd.read_csv('datas/app_train_2.csv')
    #app_train_3 = pd.read_csv('datas/app_train_3.csv')
    #app_train_4 = pd.read_csv('datas/app_train_4.csv')
    #app_train = pd.concat([app_train_1, app_train_2, app_train_3, app_train_4])
    #app_train = pd.read_csv('https://drive.google.com/file/d/16RR6zIzq2JKPMXHsT8jirXaW62jGc8W6/view?usp=drive_link')
    print("Lecture données OK")

    app_train['BIRTH_YEARS'] = app_train['DAYS_BIRTH'] / -365
    app_train = app_train.drop('DAYS_BIRTH', axis=1)

    # Create an anomalous flag column
    app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan})

    st.session_state['app_train'] = app_train
    print(app_train.shape)

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

    print('app_train encoded')
    print(app_train_enc.shape)

    data = app_train_enc.drop(columns = ['TARGET'])

    scaler_minMax = MinMaxScaler(feature_range = (0, 1))
    scaler_minMax.fit(data);

    data_scaledMM = scaler_minMax.transform(data)
    st.session_state['data_scaledMM'] = data_scaledMM

    print('data scaled')
    print(data_scaledMM.shape)

    st.session_state['model'] = mlflow.sklearn.load_model('runs:/4c4532ba1c904fcc98b806296cb62f6a/RF_full')
    print('model loaded')

    st.session_state['features_sel'] = ['EXT_SOURCE_2',
                'EXT_SOURCE_3',
                'DAYS_EMPLOYED',
                'BIRTH_YEARS',
                'AMT_CREDIT',
                'AMT_ANNUITY',
                'DAYS_LAST_PHONE_CHANGE']

def main():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

    st.markdown("## Choix du client")
    left, right = st.columns(2, vertical_alignment="bottom")

    client = left.selectbox(
        label = "Client",
        options=st.session_state['app_train']['SK_ID_CURR'].values,
        index=None,
        placeholder="Sélectionner un client",
        label_visibility="collapsed"
    )

    if right.button("Mise à jour client"):
        st.session_state['client'] =  client
        st.session_state['row'] = st.session_state['app_train'].loc[st.session_state['app_train']['SK_ID_CURR']==st.session_state['client']]

    if 'client' in st.session_state:
        print(st.session_state['client'])

        st.markdown("## Client {}".format(st.session_state['client']))
        st.session_state['index'] = np.where(st.session_state['app_train']['SK_ID_CURR']==st.session_state['client'])
        st.table(st.session_state['row'][['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE']])
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 0.5,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Score"},
            gauge = {'axis': {'range': [None, 1]},
                'threshold' : {'line': {'color': "red"}, 'value': 0.5}}))
        st.plotly_chart(fig)
        #st.session_state['model'].predict_proba(st.session_state['data_scaledMM'][st.session_state['index']].reshape(1,239))[:,1]
        

if __name__ == "__main__":
    print('start')
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        print('init')
        init()
        st.session_state.initialized = True
    
    main()