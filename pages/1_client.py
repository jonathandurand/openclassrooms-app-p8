import streamlit as st
import utils.func_cache as fc

st.markdown("# Client")
st.sidebar.markdown("# Client")

app_train = fc.app_train_load('datas/app_train_1.csv')

st.markdown("## Choix du client")
left, right = st.columns(2, vertical_alignment="bottom")

global color_array
global markdown_array
color_array = ['blue', 'red']
markdown_array = ["### :blue[Client sans risque (OK)]", "### :red[Client à risque (KO)]"]

client = left.selectbox(
    label = "Client",
    options=app_train['SK_ID_CURR'].values,
    index=None,
    placeholder="Sélectionner un client",
    label_visibility="collapsed"
)

if right.button("Mise à jour client"):
    #data = fc.data_calcul(app_train)
    data_sel_scaledMM = fc.data_sel_scaledMM_calcul(app_train[st.session_state['features_sel']])
    model, model_learn = fc.models_load()
    st.session_state['client'] =  client
    index = app_train.index[app_train['SK_ID_CURR']==st.session_state['client']]
    st.session_state['row'] = app_train.loc[app_train['SK_ID_CURR']==st.session_state['client']]
    st.session_state['row_print_client'] = fc.row_print_client(st.session_state['row'])
    st.session_state['row_print_profil'] = fc.row_print_profil(st.session_state['row'])
    st.session_state['row_sel_scaledMM'] = data_sel_scaledMM.iloc[index]
    st.session_state['score'] = model_learn.predict_proba(st.session_state['row_sel_scaledMM'])[0, 1]
    st.session_state['pred_bin'] = fc.predict_API(st.session_state['row'][st.session_state['features_sel']])
    st.session_state['color'] = color_array[st.session_state['pred_bin']]

if 'client' in st.session_state:
    #print(st.session_state['client'])
    #print(st.session_state['row'][st.session_state['features_sel']])
    #print(st.session_state['row_sel_scaledMM'])
    #print(model_learn.predict_proba(st.session_state['row_sel_scaledMM']))
    #print(st.session_state['score'])

    st.markdown("## Client {}".format(st.session_state['client']))
    st.markdown(markdown_array[st.session_state['pred_bin']])
    st.dataframe(st.session_state['row_print_client'][['Identifiant',
                                            'Genre',
                                            'Age',
                                            'Situation familiale',
                                            'Nombre d\'enfants',
                                            'Situation professionnelle']].transpose())