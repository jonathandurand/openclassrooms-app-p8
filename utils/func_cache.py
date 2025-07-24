import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import mlflow
import dagshub

import shap

from sklearn.ensemble._forest import RandomForestClassifier

import boto3
import json

global app_name
global region
app_name = 'my-deployment-attemp'
region = 'us-east-1'

@st.cache_data
def app_train_load(csv_file):
    app_train = pd.read_csv(csv_file)
    app_train['BIRTH_YEARS'] = app_train['DAYS_BIRTH'] / -365
    app_train = app_train.drop('DAYS_BIRTH', axis=1)

    #app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

    app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'].replace({365243: np.nan})

    return app_train

@st.cache_data
def data_sel_scaledMM_calcul(data):
    data_scaledMM = data.copy()

    minmax = pd.read_csv('datas/minmax_df.csv', index_col=0)
    for c in minmax.index:
        data_scaledMM[c] = (data[c]-minmax['min'][c])/(minmax['max'][c]-minmax['min'][c])

    return data_scaledMM

@st.cache_resource
def models_load():
    dagshub.auth.add_app_token('7ff59a8ec595a39c81790087b5fe632c13a71e8c')
    dagshub.init(repo_owner='jonathan.durand25', repo_name='OC_P7', mlflow=True)
    model = mlflow.pyfunc.load_model('runs:/4e1e5e9e3f3f48fda5eb52dbc836038c/model')
    model_learn = mlflow.sklearn.load_model('runs:/fa1ccfb2e0814e3d9261f7098c3d60c9/RF_sel_med')
    return model, model_learn

@st.cache_resource(hash_funcs={RandomForestClassifier: id})
def shap_values_calcul(model, row):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(row)
    return shap_vals

def query_endpoint(app_name, input_json):
	client = boto3.session.Session().client('sagemaker-runtime', region)

	response = client.invoke_endpoint(
		EndpointName = app_name,
		Body = input_json,
		ContentType = 'application/json'#'; format=pandas-split',
		)

	preds = response['Body'].read().decode('ascii')
	preds = json.loads(preds)
	#print('Received response: {}'.format(preds))
	return preds

def predict_API(data_sel):
    query_input = pd.DataFrame(data_sel).to_dict(orient='split')
    data = {"dataframe_split": query_input}
    byte_data = json.dumps(data).encode('utf-8')
    
    pred = query_endpoint(app_name=app_name, input_json=byte_data)

    pred_bin = pred['predictions'][0]

    return pred_bin

def row_print_client(row):
    row_return = row.rename(columns={'SK_ID_CURR':'Identifiant',
                                     'CODE_GENDER':'Genre',
                                     'BIRTH_YEARS':'Age',
                                     'NAME_FAMILY_STATUS':'Situation familiale',
                                     'CNT_CHILDREN':'Nombre d\'enfants',
                                     'NAME_INCOME_TYPE':'Situation professionnelle'
                                     }
                                     )
    row_return['Age'] = int(row_return['Age'])
    return row_return

def row_print_profil(row):
    row_return = pd.DataFrame()
    row_return['Source ext (2)'] =row['EXT_SOURCE_2']
    row_return['Source ext (3)'] = row['EXT_SOURCE_3']
    row_return['Emploi (a)'] = row['DAYS_EMPLOYED']/(-365)
    row_return['Age'] = row['BIRTH_YEARS']
    row_return['Total crédit (k€)'] = row['AMT_CREDIT']/1000
    row_return['Annuité (k€)'] = row['AMT_ANNUITY']/1000
    row_return['Chgt tél (a)'] = row['DAYS_LAST_PHONE_CHANGE']/(-365)

    return row_return