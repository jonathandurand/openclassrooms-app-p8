import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import plotly.graph_objects as go

import mlflow

import dagshub

import shap

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


def init():

    app_train = app_train_load('datas/app_train_1.csv')
    
    #app_train_2 = pd.read_csv('datas/app_train_2.csv')
    #app_train_3 = pd.read_csv('datas/app_train_3.csv')
    #app_train_4 = pd.read_csv('datas/app_train_4.csv')
    #app_train = pd.concat([app_train_1, app_train_2, app_train_3, app_train_4])
    print("Lecture données OK")
    print(app_train.shape)
    
    data = data_calcul(app_train)
    
    data_scaledMM = data_scaledMM_calcul(data)

    print('data scaled')
    print(data_scaledMM.shape)

    model, model_learn = models_load()

    st.session_state['threshold'] = 0.56
    print('model loaded')

    st.session_state['features_sel'] = ['EXT_SOURCE_2',
                'EXT_SOURCE_3',
                'DAYS_EMPLOYED',
                'BIRTH_YEARS',
                'AMT_CREDIT',
                'AMT_ANNUITY',
                'DAYS_LAST_PHONE_CHANGE']
    st.session_state['data_sel'] = data[st.session_state['features_sel']]
    print('data sel')

    #st.session_state['explainer'] = shap.TreeExplainer(st.session_state['model_learn'])
    #print('shap')

def main():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

    app_train = app_train_load('datas/app_train_1.csv')

    st.markdown("## Choix du client")
    left, right = st.columns(2, vertical_alignment="bottom")

    client = left.selectbox(
        label = "Client",
        options=app_train['SK_ID_CURR'].values,
        index=None,
        placeholder="Sélectionner un client",
        label_visibility="collapsed"
    )

    if right.button("Mise à jour client"):
        st.session_state['client'] =  client
        st.session_state['index'] = st.session_state['app_train'].index[st.session_state['app_train']['SK_ID_CURR']==st.session_state['client']]
        st.session_state['row'] = st.session_state['app_train'].loc[st.session_state['app_train']['SK_ID_CURR']==st.session_state['client']]
        st.session_state['row_scaledMM'] = st.session_state['data_scaledMM'].iloc[st.session_state['index']]
        st.session_state['score'] = st.session_state['model_learn'].predict_proba(st.session_state['row_scaledMM'][st.session_state['features_sel']])[0, 1]
        if st.session_state['score']<st.session_state['threshold']:
            st.session_state['color'] = 'red'
        else:
            st.session_state['color'] = 'green'

    if 'client' in st.session_state:
        print(st.session_state['client'])

        st.markdown("## Client {}".format(st.session_state['client']))
        st.session_state['index'] = np.where(st.session_state['app_train']['SK_ID_CURR']==st.session_state['client'])
        st.table(st.session_state['row'][['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE']])
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = st.session_state['score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Score"},
            gauge = {'axis': {'range': [None, 1]},
                     'bar': {'color': st.session_state['color']},
                     'threshold' : {'line': {'color': "red"}, 'value': st.session_state['threshold']}}))
        st.plotly_chart(fig)
        #st.session_state['model'].predict_proba(st.session_state['data_scaledMM'][st.session_state['index']].reshape(1,239))[:,1]
        

if __name__ == "__main__":
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        with st.spinner("Initialisation"):
            init()
            st.session_state.initialized = True
    
    main()