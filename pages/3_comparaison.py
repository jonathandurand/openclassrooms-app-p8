import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils.func_cache as fc

st.markdown("# Comparaison")
st.sidebar.markdown("# Comparaison")

app_train = fc.app_train_load('datas/app_train_1.csv')
data_sel_scaledMM = fc.data_sel_scaledMM_calcul(app_train[st.session_state['features_sel']])
model, model_learn = fc.models_load()
app_train['pred'] = fc.predict_model_tot(data_sel_scaledMM, model_learn, st.session_state['threshold'])

app_train_pos = app_train.loc[app_train['pred']==1]

left, right = st.columns(2, vertical_alignment="bottom")

with left.container():
    st.markdown("## Variable 1")
    l1, r1 = st.columns(2, vertical_alignment="bottom")

    v1 = l1.selectbox(
        label = "Variable 1",
        options=st.session_state['features_sel'],
        index=None,
        placeholder="Sélectionner une variable",
        label_visibility="collapsed"
    )

    if r1.button("Mise à jour", key="var1_update"):
        st.session_state['var1'] = v1
    
    if 'var1' in st.session_state:
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(app_train_pos[st.session_state['var1']], bins=50)
        i_max = np.max(np.where(bins<=st.session_state['row'][st.session_state['var1']].values))
        patches[i_max].set_color('red')
        ax.add_patch(patches[i_max])
        ax.set_title(st.session_state['var1']+' : densité')
        ax.set_xlabel(st.session_state['var1'])
        ax.set_ylabel('Effectif')
        st.pyplot(fig)

with right.container():
    st.markdown("## Variable 2")
    l2, r2 = st.columns(2, vertical_alignment="bottom")

    v2 = l2.selectbox(
        label = "Variable 2",
        options=st.session_state['features_sel'],
        index=None,
        placeholder="Sélectionner une variable",
        label_visibility="collapsed"
    )

    if r2.button("Mise à jour", key="var2_update"):
        st.session_state['var2'] = v2
    
    if 'var2' in st.session_state:
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(app_train_pos[st.session_state['var2']], bins=50)
        i_max = np.max(np.where(bins<=st.session_state['row'][st.session_state['var2']].values))
        patches[i_max].set_color('red')
        ax.add_patch(patches[i_max])
        ax.set_title(st.session_state['var2']+' : densité')
        ax.set_xlabel(st.session_state['var2'])
        ax.set_ylabel('Effectif')
        st.pyplot(fig)

if 'var1' in st.session_state and 'var2' in st.session_state:
    st.markdown("## Analyse bi-variée")
    fig, ax = plt.subplots()
    ax.scatter(app_train_pos[st.session_state['var1']], app_train_pos[st.session_state['var2']])
    ax.scatter(st.session_state['row'][st.session_state['var1']], st.session_state['row'][st.session_state['var2']], color='r')
    ax.set_title('Analyse bi-variée')
    ax.set_xlabel(st.session_state['var1'])
    ax.set_ylabel(st.session_state['var2'])
    st.pyplot(fig)