import streamlit as st

import utils.func_cache as fc

def init():

    app_train = fc.app_train_load('datas/app_train_1.csv')
    
    #app_train_2 = pd.read_csv('datas/app_train_2.csv')
    #app_train_3 = pd.read_csv('datas/app_train_3.csv')
    #app_train_4 = pd.read_csv('datas/app_train_4.csv')
    #app_train = pd.concat([app_train_1, app_train_2, app_train_3, app_train_4])
    print("Lecture données OK")
    print(app_train.shape)

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
    data_sel_scaledMM = fc.data_sel_scaledMM_calcul(app_train[st.session_state['features_sel']])
    print('data sel')

    app_train['pred'] = fc.predict_model_tot(data_sel_scaledMM, model_learn, st.session_state['threshold'])
    print('pred_tot')
        

if __name__ == "__main__":

    st.markdown("# Page principale")
    st.sidebar.markdown("# Page principale")

    st.write(
        """   
## Instructions d'utilisation
- Attendre l'initialisation (page principale)
- Choisir le client (page client)
- Premier affichage (page client)
    - Affichage de la prédiction de l'API en couleur
    - Affichage des caractéristiques principales du client
- Détail de la prédiction (page profil)
    - Composantes utilisées pour la prédiction
    - Détail de la probabilité
    - Importance locale (autour du client sélectionné)
    - Importance globale des variables
- Position du client dans les données prédites positives (page comparaison)
    - Choix de deux variables
    - Densité en histogramme
    - Nuage de points bi-varié
        """
    )

    if 'initialized' not in st.session_state or not st.session_state.initialized:
        with st.spinner("Initialisation (20s)", show_time=True):
            init()
        st.session_state.initialized = True
    
    st.success("Application initialisée")