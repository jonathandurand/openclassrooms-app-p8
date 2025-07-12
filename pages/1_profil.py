import streamlit as st
import streamviz
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import shap

import utils.func_cache as fc

model, model_learn = fc.models_load()

st.markdown("# Profil ❄️")
st.sidebar.markdown("# Profil ❄️")

st.markdown("## Client {}".format(st.session_state['client']))

left, right = st.columns(2, vertical_alignment="top")
left.table(st.session_state['row'][st.session_state['features_sel']].transpose())

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

plot_data = pd.DataFrame(index=st.session_state['features_sel'], data = model_learn.feature_importances_)

left, right = st.columns(2, vertical_alignment="top")

right.markdown("## Importance globale")
right.bar_chart(plot_data)

#print('sharp start')
with st.spinner("Calcul de l'importance locale"):
    shap_vals = fc.shap_values_calcul(model_learn, st.session_state['row_scaledMM'][st.session_state['features_sel']])
print(shap_vals)

plot_data2 = pd.DataFrame(data=shap_vals[:,:,1], columns=st.session_state['features_sel']).transpose()

left.markdown("## Importance locale")
left.bar_chart(plot_data2)

#print('sharp ok')
#print(shap_vals.shape)
#print(st.session_state['features_sel'])

#fig, ax = plt.subplots()
#ax = plt.gca()
#shap.bar_plot(shap_vals[0], feature_names=st.session_state['features_sel'], max_display=len(st.session_state['features_sel']), show=False);
#left.pyplot(fig)