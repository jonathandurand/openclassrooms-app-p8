import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

import utils.func_cache as fc

import shap

def init():

    app_train = fc.app_train_load('datas/app_train_1.csv')
    
    #app_train_2 = pd.read_csv('datas/app_train_2.csv')
    #app_train_3 = pd.read_csv('datas/app_train_3.csv')
    #app_train_4 = pd.read_csv('datas/app_train_4.csv')
    #app_train = pd.concat([app_train_1, app_train_2, app_train_3, app_train_4])
    print("Lecture données OK")
    print(app_train.shape)
    
    data = fc.data_calcul(app_train)
    
    data_scaledMM = fc.data_scaledMM_calcul(data)

    print('data scaled')
    print(data_scaledMM.shape)

    model, model_learn = fc.models_load()

    st.session_state['threshold'] = 0.04
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

    app_train = fc.app_train_load('datas/app_train_1.csv')

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
        data = fc.data_calcul(app_train)
        data_scaledMM = fc.data_scaledMM_calcul(data)
        model, model_learn = fc.models_load()
        st.session_state['client'] =  client
        st.session_state['index'] = app_train.index[app_train['SK_ID_CURR']==st.session_state['client']]
        st.session_state['row'] = app_train.loc[app_train['SK_ID_CURR']==st.session_state['client']]
        st.session_state['row_scaledMM'] = data_scaledMM.iloc[st.session_state['index']]
        st.session_state['score'] = model_learn.predict_proba(st.session_state['row_scaledMM'][st.session_state['features_sel']])[0, 1]
        if st.session_state['score']<st.session_state['threshold']:
            st.session_state['color'] = 'green'
        else:
            st.session_state['color'] = 'red'

    if 'client' in st.session_state:
        print(st.session_state['client'])
        print(st.session_state['row'][st.session_state['features_sel']])
        print(st.session_state['row_scaledMM'][st.session_state['features_sel']])

        st.markdown("## Client {}".format(st.session_state['client']))
        if st.session_state['color']=='green':
            st.markdown("### :green[Client sans risque]")
        else:
            st.markdown("### :red[Client à risque]")
        st.table(st.session_state['row'][['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE']])
        
        #st.session_state['model'].predict_proba(st.session_state['data_scaledMM'][st.session_state['index']].reshape(1,239))[:,1]
        

if __name__ == "__main__":
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        init()
        st.session_state.initialized = True
    
    main()