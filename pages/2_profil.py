import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import utils.func_cache as fc

model, model_learn = fc.models_load()

st.markdown("# Profil")
st.sidebar.markdown("# Profil")


st.markdown("## Client {}".format(st.session_state['client']))
left, right = st.columns(2, vertical_alignment="top")

left.dataframe(st.session_state['row_print_profil'].transpose().style.format(precision=1))

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = st.session_state['score'],
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Score"},
    gauge = {'axis': {'range': [None, 1]},
             'bar': {'color': st.session_state['color']},
             'threshold' : {'line': {'color': "red"}, 'value': st.session_state['threshold']}}))
#fig.update_layout(autosize=False, height=300)
right.plotly_chart(fig)

    
st.markdown("## Facteurs de prédiction")
left, right = st.columns(2, vertical_alignment="top")

right.markdown("### Importance globale")
plot_data = pd.DataFrame(index=st.session_state['features_sel'], data = model_learn.feature_importances_)
right.bar_chart(plot_data)

left.markdown("### Importance locale")
with st.spinner("Calcul de l'importance locale"):
    shap_vals = fc.shap_values_calcul(model_learn, st.session_state['row_sel_scaledMM'])


plot_data2 = pd.DataFrame(data=shap_vals[:,:,1], columns=st.session_state['features_sel']).transpose()
left.bar_chart(plot_data2)