import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("Smile & Pay – Rapport BI & Agent Conversationnel")

# --- 1. Upload des 3 fichiers ---
st.sidebar.header("Chargement des fichiers")
file_tx = st.sidebar.file_uploader("Transactions (Test)", type=["csv", "xlsx"], key="tx")
file_merch = st.sidebar.file_uploader("Caractéristiques marchands (Deals Insights)", type=["csv", "xlsx"], key="merch")
file_weather = st.sidebar.file_uploader("Données météo (Synop Essentials)", type=["csv", "xlsx"], key="weather")

# Lecture tolérante
@st.cache_data
def read_file(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith('.csv'):
        try:
            return pd.read_csv(uploaded, sep=None, engine='python')
        except:
            return pd.read_csv(uploaded, sep=',')
    else:
        return pd.read_excel(uploaded, engine='openpyxl')

df_tx = read_file(file_tx)
df_merch = read_file(file_merch)
df_weather = read_file(file_weather)

if df_tx is not None and df_merch is not None and df_weather is not None:
    st.success("✅ Tous les fichiers chargés")

    # --- 2. Préparation et fusion ---
    # Nettoyage montant
    df_tx['MONTANT'] = df_tx['MONTANT'].astype(str).str.replace(r"[^0-9,.-]", '', regex=True).str.replace(',', '.').astype(float)
    # DATETIME
    df_tx['DATETIME'] = pd.to_datetime(df_tx['DATE'].astype(str) + ' ' + df_tx['HEURE'].astype(str), dayfirst=True, errors='coerce')
    df_tx['HOUR'] = df_tx['DATETIME'].dt.hour
    df_tx['DAY'] = df_tx['DATETIME'].dt.date
    df_tx['WEEK'] = df_tx['DATETIME'].dt.to_period('W').apply(lambda r: r.start_time)
    df_tx['MONTH'] = df_tx['DATETIME'].dt.to_period('M').apply(lambda r: r.start_time)

    # Fusion Merchants
    df = df_tx.merge(df_merch[['REF_MARCHAND', 'Organization_type']], on='REF_MARCHAND', how='left')
    df.rename(columns={'Organization_type': 'TYPE_COMMERCE'}, inplace=True)
    # Fusion Weather
    df = df.merge(df_weather[['CODE POSTAL', 'Température']], on='CODE POSTAL', how='left')

    # --- 3. Rapport BI détaillé ---
    if st.button("📄 Générer rapport BI détaillé"):
        st.header("1. Indicateurs descriptifs")
        # Volume total
        total_tx = len(df)
        st.metric("Transactions totales", f"{total_tx:,}")
        
        # Par période
        cols = st.columns(3)
        cols[0].write("**Journée**")
        by_day = df.groupby('DAY').size()
        cols[0].write(by_day)
        cols[1].write("**Semaine**")
        by_week = df.groupby('WEEK').size()
        cols[1].write(by_week)
        cols[2].write("**Mois**")
        by_month = df.groupby('MONTH').size()
        cols[2].write(by_month)

        # Evolution périodique
        st.subheader("Évolution périodique (T/T-1)")
        evo = by_month.pct_change().fillna(0)
        st.line_chart(evo)

        # Montant total et panier moyen
        ca_total = df['MONTANT'].sum()
        panier_moy = ca_total / total_tx
        st.metric("CA total (€)", f"{ca_total:,.2f}")
        st.metric("Panier moyen (€)", f"{panier_moy:,.2f}")

        # Distribution des montants
        st.subheader("Distribution des montants")
        desc = df['MONTANT'].describe(percentiles=[0.1,0.25,0.5,0.75,0.9])
        st.table(desc[['10%', '25%', '50%', '75%', '90%', 'mean']].rename({'10%':'P10','25%':'P25','50%':'Médiane','75%':'P75','90%':'P90','mean':'Moyenne'}))

        # Répartition par type de commerce
        st.header("Répartition par type de commerce")
        top = df['TYPE_COMMERCE'].value_counts()
        st.bar_chart(top)
        ca_type = df.groupby('TYPE_COMMERCE')['MONTANT'].agg(['sum','count'])
        ca_type['panier_moy'] = ca_type['sum']/ca_type['count']
        st.dataframe(ca_type.sort_values('count', ascending=False))

        # Analyse spatiale
        st.header("Analyse spatiale par code postal")
        by_cp = df.groupby('CODE POSTAL').agg(tx_count=('MONTANT','size'), ca=('MONTANT','sum'))
        by_cp['panier_moy'] = by_cp['ca']/by_cp['tx_count']
        st.dataframe(by_cp)
        # TODO: heatmap géographique

        # Temporalité fine
        st.header("Temporalité fine")
        # Horaire
        hourly = df.groupby('HOUR')['MONTANT'].agg(['count','mean'])
        fig, ax = plt.subplots(figsize=(6,3))
        hourly['count'].plot.bar(ax=ax)
        ax.set_title('Transactions par heure')
        st.pyplot(fig)
        # Hebdo vs Week-end
        df['WEEKEND'] = df['DATETIME'].dt.dayofweek >= 5
        wb = df.groupby('WEEKEND').size()
        st.write({'Semaine': wb[False], 'Week-end': wb[True]})
        # Saisonnalité
        st.line_chart(by_month)

        st.info("Les sections diagnostics et prédictives sont en cours d’implémentation.")
else:
    st.warning("Veuillez charger les 3 fichiers Excel pour continuer.")

# --- Agent Conversationnel ---
# ... (inchangé)
