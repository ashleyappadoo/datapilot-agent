import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pgeocode
import pydeck as pdk
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

    # Normalisation colonnes météo
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

    # Fusion
    df = (
        df_tx
        .merge(
            df_merch.rename(columns={'Organization_type': 'TYPE_COMMERCE'})[['REF_MARCHAND', 'TYPE_COMMERCE']],
            on='REF_MARCHAND', how='left'
        )
        .merge(df_weather[['CODE POSTAL', 'TEMP']], on='CODE POSTAL', how='left')
    )

    # Géocodage des codes postaux FR pour chaque code unique
    nomi = pgeocode.Nominatim('fr')
    unique_cp = df[['CODE POSTAL']].drop_duplicates().astype(str)
    geo = nomi.query_postal_code(unique_cp['CODE POSTAL'])
    df_geo = geo[['postal_code', 'latitude', 'longitude']].rename(columns={'postal_code': 'CODE POSTAL'}).dropna()

    # --- 3. Rapport BI détaillé ---
    if st.button("📄 Générer rapport BI détaillé"):
        st.header("1. Indicateurs descriptifs")
        total_tx = len(df)
        st.metric("Transactions totales", f"{total_tx:,}")
        n_days = df['DAY'].nunique()
        by_day = df.groupby('DAY').size()
        by_week = df.groupby('WEEK').size()
        by_month = df.groupby('MONTH').size()
        if n_days == 1:
            day0 = by_day.index[0]
            st.subheader(f"Analyse pour {day0}")
            st.write(int(by_day.iloc[0]))
        else:
            cols = st.columns(3)
            cols[0].write(by_day)
            cols[1].write(by_week)
            cols[2].write(by_month)
        if by_month.size > 1:
            st.subheader("Évolution périodique (T/T-1)")
            st.line_chart(by_month.pct_change().fillna(0))

        ca_total = df['MONTANT'].sum()
        panier_moy = ca_total / total_tx
        st.metric("CA total (€)", f"{ca_total:,.2f}")
        st.metric("Panier moyen (€)", f"{panier_moy:,.2f}")

        st.subheader("Distribution des montants")
        desc = df['MONTANT'].describe(percentiles=[0.1,0.25,0.5,0.75,0.9])
        st.table(desc[['10%','25%','50%','75%','90%','mean']]
                  .rename({'10%':'P10','25%':'P25','50%':'Médiane','75%':'P75','90%':'P90','mean':'Moyenne'}))

        st.header("Répartition par type de commerce")
        st.bar_chart(df['TYPE_COMMERCE'].value_counts())
        ca_type = df.groupby('TYPE_COMMERCE')['MONTANT'].agg(['sum','count'])
        ca_type['panier_moy'] = ca_type['sum']/ca_type['count']
        st.dataframe(ca_type.sort_values('count', ascending=False))

        # Spatial descriptif avec agrégation par code postal
        st.header("Analyse spatiale par code postal")
        by_cp = df.groupby('CODE POSTAL').agg(
            tx_count=('MONTANT','size'),
            ca=('MONTANT','sum')
        )
        by_cp['panier_moy'] = by_cp['ca']/by_cp['tx_count']
        # Fusion avec géolocalisation
        by_cp = by_cp.reset_index().merge(df_geo, on='CODE POSTAL', how='left').dropna(subset=['latitude','longitude'])
        st.dataframe(by_cp)

        # Carte géographique : panier moyen
        st.subheader("Carte géographique : panier moyen par code postal")
        deck1 = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v10',
            initial_view_state=pdk.ViewState(latitude=46.5, longitude=2.5, zoom=5),
            layers=[
                pdk.Layer(
                    'HeatmapLayer',
                    data=by_cp,
                    get_position='[longitude, latitude]',
                    get_weight='panier_moy',
                    radiusPixels=50
                )
            ]
        )
        st.pydeck_chart(deck1)

        # Carte géographique : montant total
        st.subheader("Carte géographique : montant total par code postal")
        deck2 = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v10',
            initial_view_state=pdk.ViewState(latitude=46.5, longitude=2.5, zoom=5),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=by_cp,
                    get_position='[longitude, latitude]',
                    get_radius='ca / tx_count * 2000',
                    get_fill_color='[200, 30, 0, 160]',
                    pickable=True
                )
            ]
        )
        st.pydeck_chart(deck2)

        # Diagnostics
        st.header("2. Indicateurs diagnostics")
        st.subheader("Corrélation température vs montant")
        corr = df[['TEMP','MONTANT']].corr().loc['TEMP','MONTANT']
        st.write(f"Coefficient de corrélation (Pearson) : {corr:.2f}")
        fig, ax = plt.subplots()
        ax.scatter(df['TEMP'], df['MONTANT'], alpha=0.3)
        coef = np.polyfit(df['TEMP'], df['MONTANT'], 1)
        ax.plot(df['TEMP'], coef[0]*df['TEMP']+coef[1], color='red')
        ax.set_xlabel('Température (°C)'); ax.set_ylabel('Montant (€)')
        st.pyplot(fig)

        st.subheader("Panier moyen par classes de température")
        bins = [-np.inf,5,15,np.inf]
        labels = ['<5°C','5-15°C','>15°C']
        df['TEMP_BINS'] = pd.cut(df['TEMP'], bins=bins, labels=labels)
        tb = df.groupby('TEMP_BINS')['MONTANT'].mean()
        st.bar_chart(tb)

        st.subheader("Sensibilité du panier moyen à 1°C")
        lr = LinearRegression().fit(df[['TEMP']], df['MONTANT'])
        st.write(f"Variation moyenne du panier par °C : {lr.coef_[0]:.2f} €")

        st.header("3. Segmentation clients")
        feats = df[['MONTANT','HOUR','TEMP']].dropna()
        scaler = StandardScaler().fit(feats)
        X = scaler.transform(feats)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        df['cluster'] = kmeans.labels_
        st.subheader("Répartition des clusters (KMeans)")
        st.bar_chart(df['cluster'].value_counts().sort_index())
        prof = df.groupby('cluster').agg({
            'MONTANT':'mean', 'HOUR':'mean', 'TEMP':'mean', 'REF_MARCHAND':'count'
        }).rename(columns={'REF_MARCHAND':'nb_tx'})
        st.dataframe(prof)

        st.info("Sections prédictives à venir.")
else:
    st.warning("Veuillez charger les 3 fichiers Excel pour continuer.")
