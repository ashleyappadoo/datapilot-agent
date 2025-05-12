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
    if not u: return None
    if u.name.lower().endswith('.csv'):
        try: return pd.read_csv(u, sep=None, engine='python')
        except: return pd.read_csv(u)
    return pd.read_excel(u, engine='openpyxl')

df_tx = read_file(file_tx)
df_merch = read_file(file_merch)
df_weather = read_file(file_weather)

if df_tx is not None and df_merch is not None and df_weather is not None:
    st.success("✅ Tous les fichiers chargés")

    # --- 2. Préparation et fusion ---
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

    df_weather.columns = (
        df_weather.columns
                  .str.strip()
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.upper()
    )
    if 'TEMPÉRATURE' in df_weather.columns:
        df_weather.rename(columns={'TEMPÉRATURE': 'TEMP'}, inplace=True)
    if 'CODE POSTAL' not in df_weather.columns and 'CODE_POSTAL' in df_weather.columns:
        df_weather.rename(columns={'CODE_POSTAL': 'CODE POSTAL'}, inplace=True)

    df = (
        df_tx
        .merge(df_merch.rename(columns={'Organization_type': 'TYPE_COMMERCE'})[['REF_MARCHAND', 'TYPE_COMMERCE']],
               on='REF_MARCHAND', how='left')
        .merge(df_weather[['CODE POSTAL', 'TEMP']], on='CODE POSTAL', how='left')
    )

    nomi = pgeocode.Nominatim('fr')
    unique_cp = df['CODE POSTAL'].astype(str).str.zfill(5).drop_duplicates().tolist()
    geo = nomi.query_postal_code(unique_cp)
    df_geo = geo[['postal_code','latitude','longitude']].dropna()
    df_geo['postal_code'] = df_geo['postal_code'].astype(str).str.zfill(5)
    df_geo.rename(columns={'postal_code':'CODE POSTAL'}, inplace=True)

    # --- 3. Rapport BI détaillé ---
    if st.button("📄 Générer rapport BI détaillé"):
        # 1. INDICATEURS DESCRIPTIFS
        st.header("1. Indicateurs descriptifs")
        total_tx = len(df)
        st.metric("Transactions totales", f"{total_tx:,}")

        # Périodes
        by_day = df.groupby('DAY').size()
        by_week = df.groupby('WEEK').size()
        by_month = df.groupby('MONTH').size()
        if by_day.size == 1:
            st.subheader(f"Analyse journalière ({by_day.index[0]})")
            st.write(int(by_day.iloc[0]))
        else:
            cols = st.columns(3)
            cols[0].write("**Transactions par jour**")
            cols[0].write(by_day)
            cols[1].write("**Transactions par semaine**")
            cols[1].write(by_week)
            cols[2].write("**Transactions par mois**")
            cols[2].write(by_month)
        if by_month.size > 1:
            st.subheader("Évolution périodique (T/T-1)")
            st.line_chart(by_month.pct_change().fillna(0))

        # CA & panier
        ca_total = df['MONTANT'].sum()
        panier_moy = ca_total / total_tx
        st.metric("Chiffre d'affaires total (€)", f"{ca_total:,.2f}")
        st.metric("Panier moyen (€)", f"{panier_moy:,.2f}")

        # Distribution montants
        stats = df['MONTANT'].describe(percentiles=[0.1,0.25,0.5,0.75,0.9])
        st.subheader("Distribution des montants")
        st.table(stats[['10%','25%','50%','75%','90%','mean']]
                  .rename({'10%':'P10','25%':'P25','50%':'Médiane','75%':'P75','90%':'P90','mean':'Moyenne'}))

        # Répartition par type
        st.subheader("Répartition par type de commerce")
        part = df['TYPE_COMMERCE'].value_counts()
        st.bar_chart(part)
        ca_type = df.groupby('TYPE_COMMERCE')['MONTANT'].agg(['sum','count'])
        ca_type['panier_moy'] = ca_type['sum'] / ca_type['count']
        st.dataframe(ca_type.sort_values('count', ascending=False))

        # Top-performers
        st.subheader("Top 5 marchands par nombre de transactions")
        top_tx = df['MARCHAND'].value_counts().head(5)
        st.write(top_tx)
        st.subheader("Top 5 marchands par CA")
        top_ca = df.groupby('MARCHAND')['MONTANT'].sum().nlargest(5)
        st.write(top_ca)

        # 2. TEMPORALITÉ FINE
        st.header("Temporalité fine")
        hourly = df.groupby('HOUR').agg(nb=('MONTANT','size'), avg=('MONTANT','mean'))
        fig1, ax1 = plt.subplots()
        hourly['nb'].plot(kind='bar', ax=ax1, title='Transactions par tranche horaire')
        ax1.set_xlabel('Heure'); ax1.set_ylabel('Nb TX')
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots()
        hourly['avg'].plot(kind='bar', ax=ax2, title='Panier moyen par heure')
        ax2.set_xlabel('Heure'); ax2.set_ylabel('Panier moyen (€)')
        st.pyplot(fig2)
        # Semaine vs week-end
        df['WEEKEND'] = df['DATETIME'].dt.dayofweek >= 5
        wk = df['WEEKEND'].value_counts().rename({False:'Semaine', True:'Week-end'})
        st.subheader("Transactions semaine vs week-end")
        st.bar_chart(wk)
        # Saisonnalité
        if by_month.size > 1:
            st.subheader("Variations mensuelles")
            st.line_chart(by_month)

        # 3. ANALYSE SPATIALE
        st.header("Analyse spatiale par code postal")
        spatial = df.groupby('CODE POSTAL').agg(
            tx_count=('MONTANT','size'), ca=('MONTANT','sum')
        ).reset_index()
        spatial['panier_moy'] = spatial['ca'] / spatial['tx_count']
        spatial = spatial.merge(df_geo, on='CODE POSTAL', how='left')
        st.dataframe(spatial)
        map_data = spatial.dropna(subset=['latitude','longitude'])
        if not map_data.empty:
            st.subheader("Heatmap panier moyen")
            deck1 = pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v10',
                initial_view_state=pdk.ViewState(latitude=46.5, longitude=2.5, zoom=5),
                layers=[pdk.Layer('HeatmapLayer', data=map_data,
                                  get_position=['longitude','latitude'], get_weight='panier_moy', radiusPixels=50)]
            )
            st.pydeck_chart(deck1)
            st.subheader("Scatter : montant total")
            deck2 = pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v10',
                initial_view_state=pdk.ViewState(latitude=46.5, longitude=2.5, zoom=5),
                layers=[pdk.Layer('ScatterplotLayer', data=map_data,
                                  get_position=['longitude','latitude'], get_radius='tx_count', radius_scale=200,
                                  get_fill_color=[200,30,0,160], pickable=True)]
            )
            st.pydeck_chart(deck2)

        # 4. INDICATEURS DIAGNOSTICS
        st.header("Corrélation et diagnostics")
        # Corrélation température vs montant
        corr = df[['TEMP','MONTANT']].corr().loc['TEMP','MONTANT']
        st.write(f"Coefficient Pearson température vs montant : {corr:.2f}")
        fig3, ax3 = plt.subplots()
        ax3.scatter(df['TEMP'], df['MONTANT'], alpha=0.3)
        # Droite de tendance
        valid = df[['TEMP','MONTANT']].dropna()
        if len(valid) > 1:
            coef = np.polyfit(valid['TEMP'], valid['MONTANT'], 1)
            ax3.plot(valid['TEMP'], coef[0]*valid['TEMP']+coef[1], color='red')
        ax3.set_xlabel('Température (°C)'); ax3.set_ylabel('Montant (€)')
        st.pyplot(fig3)
        # Panier moyen par classe météo
        bins = [-np.inf,5,15,np.inf]; labels=['<5°C','5-15°C','>15°C']
        df['TEMP_BIN'] = pd.cut(df['TEMP'], bins=bins, labels=labels)
        tb = df.groupby('TEMP_BIN')['MONTANT'].mean()
        st.bar_chart(tb)
        # Sensibilité panier/°C
        if len(valid) > 1:
            lr = LinearRegression().fit(valid[['TEMP']], valid['MONTANT'])
            st.write(f"Élasticité (€/°C) : {lr.coef_[0]:.2f}")
        # Outliers
        q1, q3 = df['MONTANT'].quantile([0.25,0.75])
        iqr = q3 - q1
        out = df[(df['MONTANT'] < q1-1.5*iqr) | (df['MONTANT'] > q3+1.5*iqr)]
        st.subheader("Comportements atypiques (outliers)")
        st.write(f"Total outliers : {len(out)}")
        if 'TYPE_COMMERCE' in out:
            st.write(out['TYPE_COMMERCE'].value_counts())

        # 5. SEGMENTATION CLIENTÈLE
        st.header("Segmentation client")
        feats = df[['MONTANT','HOUR','TEMP']].dropna()
        if len(feats) >= 3:
            # KMeans
            scaler = StandardScaler().fit(feats)
            X = scaler.transform(feats)
            km = KMeans(n_clusters=3, random_state=42).fit(X)
            df.loc[feats.index,'cluster_km'] = km.labels_
            st.subheader("Clusters KMeans")
            st.bar_chart(df['cluster_km'].value_counts())
            # DBSCAN
            db = DBSCAN(eps=0.5, min_samples=5).fit(X)
            df.loc[feats.index,'cluster_db'] = db.labels_
            st.subheader("Clusters DBSCAN")
            st.bar_chart(df['cluster_db'].value_counts())
            # Profils
            prof = df.groupby('cluster_km').agg({'MONTANT':'mean','HOUR':'mean','TEMP':'mean','REF_MARCHAND':'count'})
            prof.rename(columns={'REF_MARCHAND':'nb_tx'}, inplace=True)
            st.dataframe(prof)
        else:
            st.write("⚠️ Pas assez de données pour segmentation.")

        st.info("Sections prédictives (forecasting, alerting) à venir.")

     # --- 2. Définition des fonctions exposées au LLM ---
    def get_mean_by_type():
        data = df.groupby('TYPE_COMMERCE')['MONTANT'].mean().to_dict()
        return {'mean_by_type': data}

    functions = [
        {
            'name': 'get_mean_by_type',
            'description': 'Get average transaction amount per commerce type',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            }
        }
    ]

    # --- 3. Agent conversationnel avec function calling ---
    st.header("💬 Interrogez l'agent BI")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Votre question :", key='chat_input')
    if st.button("Envoyer", key='send_btn') and user_input:
        # Assemble messages
        messages = [
            {'role':'system','content':'Tu es un assistant BI expert. Utilise les fonctions si nécessaire.'},
            *st.session_state.chat_history,
            {'role':'user','content':user_input}
        ]
        # Call with function definitions
        response = openai.chat.completions.create(
            model='gpt-4-0613',
            messages=messages,
            functions=functions,
            function_call='auto'
        )
        msg = response.choices[0].message
        # If function call requested
        if msg.get('function_call'):
            fn_name = msg['function_call']['name']
            fn_args = json.loads(msg['function_call']['arguments'] or '{}')
            fn_response = globals()[fn_name](**fn_args)
            # Add function response to history
            st.session_state.chat_history.append({'role':'assistant','content':f'Appel de fonction: {fn_name}'} )
            st.session_state.chat_history.append({'role':'function','name':fn_name,'content':json.dumps(fn_response)})
            # Second call to get final answer
            messages.append({'role':'assistant','content':None,'function_call':msg['function_call']})
            messages.append({'role':'function','name':fn_name,'content':json.dumps(fn_response)})
            second = openai.chat.completions.create(
                model='gpt-4-0613',
                messages=messages
            )
            reply = second.choices[0].message.content
        else:
            reply = msg.content
        # Store and display
        st.session_state.chat_history.append({'role':'user','content':user_input})
        st.session_state.chat_history.append({'role':'assistant','content':reply})

    # Render chat history
    for chat in st.session_state.chat_history:
        icon = '👤' if chat['role']=='user' else '🤖'
        content = chat.get('content')
        st.markdown(f"**{icon}** {content}")
else:
    st.warning("Chargez d'abord les 3 fichiers Excel.")

