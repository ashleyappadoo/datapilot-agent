import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pgeocode
import pydeck as pdk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
import json
import openai

st.set_page_config(layout="wide")
st.title("Smile Agent – Rapport BI & Agent Conversationnel")

# --- 1. Upload des 3 fichiers ---
st.sidebar.header("Chargement des fichiers")
file_tx = st.sidebar.file_uploader("Transactions (Test)", ["csv","xlsx"], key="tx")
file_merch = st.sidebar.file_uploader("Caractéristiques marchands", ["csv","xlsx"], key="merch")
file_weather = st.sidebar.file_uploader("Données météo", ["csv","xlsx"], key="weather")

@st.cache_data
def read_file(u):
    if u is None:
        return None
    if u.name.lower().endswith('.csv'):
        try:
            return pd.read_csv(u, sep=None, engine='python')
        except:
            return pd.read_csv(u)
    return pd.read_excel(u, engine='openpyxl')

# Lecture des fichiers
df_tx = read_file(file_tx)
df_merch = read_file(file_merch)
df_weather = read_file(file_weather)

# Vérification
if df_tx is not None and df_merch is not None and df_weather is not None:
    st.success("✅ Tous les fichiers chargés")

    # --- 2. Préparation et fusion ---
    # Nettoyage et datetime
    df_tx['MONTANT'] = (
        df_tx['MONTANT'].astype(str)
              .str.replace(r"[^0-9,.-]", '', regex=True)
              .str.replace(',', '.')
              .astype(float)
    )
    df_tx['DATETIME'] = pd.to_datetime(
        df_tx['DATE'].astype(str) + ' ' + df_tx['HEURE'].astype(str),
        dayfirst=True, errors='coerce'
    )
    df_tx['HOUR'] = df_tx['DATETIME'].dt.hour
    df_tx['DAY'] = df_tx['DATETIME'].dt.date
    df_tx['WEEK'] = df_tx['DATETIME'].dt.to_period('W').apply(lambda r: r.start_time)
    df_tx['MONTH'] = df_tx['DATETIME'].dt.to_period('M').apply(lambda r: r.start_time)

    # Normalisation météo
    df_weather.columns = (
        df_weather.columns
                  .str.strip()
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.upper()
    )
    if 'TEMPÉRATURE' in df_weather.columns:
        df_weather.rename(columns={'TEMPÉRATURE': 'TEMP'}, inplace=True)
    if 'CODE_POSTAL' in df_weather.columns and 'CODE POSTAL' not in df_weather.columns:
        df_weather.rename(columns={'CODE_POSTAL': 'CODE POSTAL'}, inplace=True)

    # Fusion transactions, marchands, météo
    df = (
        df_tx
        .merge(
            df_merch.rename(columns={'Organization_type': 'TYPE_COMMERCE'})[['REF_MARCHAND', 'TYPE_COMMERCE']],
            on='REF_MARCHAND', how='left'
        )
        .merge(df_weather[['CODE POSTAL', 'TEMP']], on='CODE POSTAL', how='left')
    )

    # Géocodage
    nomi = pgeocode.Nominatim('fr')
    unique_cp = df['CODE POSTAL'].astype(str).str.zfill(5).drop_duplicates().tolist()
    geo = nomi.query_postal_code(unique_cp)
    df_geo = geo[['postal_code', 'latitude', 'longitude']].dropna()
    df_geo['postal_code'] = df_geo['postal_code'].astype(str).str.zfill(5)
    df_geo.rename(columns={'postal_code': 'CODE POSTAL'}, inplace=True)

    # --- 3. Rapport BI détaillé ---
    if st.button("📄 Générer rapport BI détaillé"):
        # ... Ton code existant pour indicateurs descriptifs, spatial, diagnostics, segmentation ...
        st.info("Sections prédictives à venir.")

    # --- Fonctions exposées au LLM ---
    def get_mean_by_type():
        return {'mean_by_type': df.groupby('TYPE_COMMERCE')['MONTANT'].mean().to_dict()}

    def get_top_merchants(by='transactions', top_n=5):
        if by == 'transactions':
            data = df['MARCHAND'].value_counts().head(top_n).to_dict()
        else:
            data = df.groupby('MARCHAND')['MONTANT'].sum().nlargest(top_n).to_dict()
        return {'top_merchants': data}

    def get_correlation():
        corr = df[['TEMP', 'MONTANT']].corr().loc['TEMP', 'MONTANT']
        return {'correlation_temp_amount': float(corr)}

    def get_distribution(percentiles=[0.1,0.25,0.5,0.75,0.9]):
        desc = df['MONTANT'].describe(percentiles=percentiles)
        result = {str(p): float(desc[f"{int(p*100)}%"] ) for p in percentiles}
        result['mean'] = float(desc['mean'])
        return {'distribution': result}

    def get_tpe_count_by_type():
        data = df.groupby(['TYPE_COMMERCE', 'MODELE_TERMINAL']).size().reset_index(name='count')
        records = data.to_dict(orient='records')
        return {'tpe_count_by_type': records}

    functions = [
        {'name': 'get_mean_by_type', 'description': 'Panier moyen par type de commerce', 'parameters': {'type':'object','properties':{},'required':[]}},
        {'name': 'get_top_merchants', 'description': 'Top marchands par transactions ou CA', 'parameters': {'type':'object','properties':{'by':{'type':'string','enum':['transactions','revenue']},'top_n':{'type':'integer'}},'required':[]}},
        {'name': 'get_correlation', 'description': 'Corrélation Pearson température vs montant', 'parameters': {'type':'object','properties':{},'required':[]}},
        {'name': 'get_distribution', 'description': 'Distribution des montants aux percentiles donnés', 'parameters': {'type':'object','properties':{'percentiles':{'type':'array','items':{'type':'number'}}},'required':[]}},
        {'name': 'get_tpe_count_by_type', 'description': 'Nombre de transactions par MODELE_TERMINAL et type de commerce', 'parameters': {'type':'object','properties':{},'required':[]}}
    ]

    # --- Agent conversationnel ---
    st.header("💬 Interrogez l'agent BI")
    # Initialisation de l'historique
    if 'chat_history' not in st.session_state or not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []

    # Saisie utilisateur
    user_input = st.text_input("Votre question :", key='chat_input')
    if st.button("Envoyer", key='send_btn') and user_input:
        st.session_state.chat_history.append({'role':'user','content':user_input})
        # Construction du prompt
        messages = [{'role':'system','content':'Tu es un assistant BI expert. Utilise les outils disponibles.'}]
        messages += st.session_state.chat_history
        # Appel OpenAI avec function_call
        response = openai.chat.completions.create(
            model='gpt-4-0613',
            messages=messages,
            functions=functions,
            function_call='auto'
        )
        msg = response.choices[0].message
        # Exécution éventuelle d'une fonction
        if getattr(msg, 'function_call', None):
            fn = msg.function_call.name
            args = json.loads(msg.function_call.arguments or '{}')
            result = globals()[fn](**args)
            # Ajout contexte fonction
            st.session_state.chat_history.append({'role':'function','name':fn,'content':json.dumps(result)})
            # Second appel pour formuler réponse
            follow = openai.chat.completions.create(
                model='gpt-4-0613',
                messages=[
                    *messages,
                    {'role':'assistant','content':None,'function_call':msg.function_call},
                    {'role':'function','name':fn,'content':json.dumps(result)}
                ]
            )
            reply = follow.choices[0].message.content
        else:
            reply = msg.content
        st.session_state.chat_history.append({'role':'assistant','content':reply})

    # Affichage historique uniquement user/assistant
    for chat in st.session_state.chat_history:
        if chat['role'] in ['user','assistant']:
            icon = '👤' if chat['role']=='user' else '🤖'
            st.markdown(f"**{icon}** {chat['content']}")

else:
    st.warning("Chargez d'abord les 3 fichiers Excel.")
