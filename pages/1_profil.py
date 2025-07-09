import streamlit as st
import streamviz
import pandas as pd

st.markdown("# Profil ❄️")
st.sidebar.markdown("# Profil ❄️")

st.markdown("## Client {}".format(st.session_state['client']))

left, right = st.columns(2, vertical_alignment="bottom")
left.table(st.session_state['row'][st.session_state['features_sel']].transpose())

with right.container():
    streamviz.gauge(
            0.5,
            sFix="%",
            gSize="MED"
            )
    
st.markdown("## Facteurs de prédiction")

plot_data = pd.DataFrame(index=st.session_state['features_sel'], data = st.session_state['model'].feature_importances_)

left, right = st.columns(2, vertical_alignment="bottom")

right.bar_chart(plot_data)