import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pgeocode
import pydeck as pdk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
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
        .merge(df_merch.rename(columns={'Organization_type': 'TYPE_COMMERCE'})[['REF_MARCHAND', 'TYPE_COMMERCE']], on='REF_MARCHAND', how='left')
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
            cols[0].write("**Transactions par jour**"); cols[0].write(by_day)
            cols[1].write("**Transactions par semaine**"); cols[1].write(by_week)
            cols[2].write("**Transactions par mois**"); cols[2].write(by_month)
        if by_month.size > 1:
            st.subheader("Évolution périodique (T/T-1)")
            st.line_chart(by_month.pct_change().fillna(0))

        # CA & panier
        ca_total = df['MONTANT'].sum(); panier_moy = ca_total/total_tx
        st.metric("CA total (€)", f"{ca_total:,.2f}"); st.metric("Panier moyen (€)", f"{panier_moy:,.2f}")

        # Distribution montants
        desc = df['MONTANT'].describe(percentiles=[0.1,0.25,0.5,0.75,0.9])
        st.subheader("Distribution des montants")
        st.table(desc[['10%','25%','50%','75%','90%','mean']]
                  .rename({'10%':'P10','25%':'P25','50%':'Médiane','75%':'P75','90%':'P90','mean':'Moyenne'}))

        # Répartition par type
        st.subheader("Répartition par type de commerce")
        st.bar_chart(df['TYPE_COMMERCE'].value_counts())
        ca_type = df.groupby('TYPE_COMMERCE')['MONTANT'].agg(['sum','count'])
        ca_type['panier_moy'] = ca_type['sum']/ca_type['count']
        st.dataframe(ca_type.sort_values('count', ascending=False))

        # Top-performers
        st.subheader("Top 5 marchands par volume")
        st.write(df['MARCHAND'].value_counts().head(5))
        st.subheader("Top 5 marchands par CA")
        st.write(df.groupby('MARCHAND')['MONTANT'].sum().nlargest(5))

        # Temporalité fine
        st.header("Temporalité fine")
        hourly = df.groupby('HOUR').agg(nb=('MONTANT','size'), avg=('MONTANT','mean'))
        fig, ax = plt.subplots(); hourly['nb'].plot.bar(ax=ax); ax.set(title='TX/heure', xlabel='Heure'); st.pyplot(fig)
        fig, ax = plt.subplots(); hourly['avg'].plot.bar(ax=ax); ax.set(title='Panier moyen/heure'); st.pyplot(fig)
        df['WEEKEND'] = df['DATETIME'].dt.dayofweek >=5
        wk = df['WEEKEND'].value_counts().rename({False:'Semaine',True:'Week-end'})
        st.subheader("Semaine vs Week-end"); st.bar_chart(wk)
        if by_month.size>1:
            st.subheader("Variations mensuelles"); st.line_chart(by_month)

        # Analyse spatiale
        st.header("Analyse spatiale par code postal")
        spatial = df.groupby('CODE POSTAL').agg(tx=('MONTANT','size'), ca=('MONTANT','sum')).reset_index()
        spatial['panier_moy']=spatial['ca']/spatial['tx']
        spatial=spatial.merge(df_geo,on='CODE POSTAL',how='left')
        st.dataframe(spatial)
        map_data=spatial.dropna(subset=['latitude','longitude'])
        if not map_data.empty:
            st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v10',initial_view_state=pdk.ViewState(46.5,2.5,5),layers=[pdk.Layer('HeatmapLayer',data=map_data,get_position=['longitude','latitude'],get_weight='panier_moy',radiusPixels=50)]))
            st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v10',initial_view_state=pdk.ViewState(46.5,2.5,5),layers=[pdk.Layer('ScatterplotLayer',data=map_data,get_position=['longitude','latitude'],get_radius='tx',radius_scale=200,get_fill_color=[200,30,0,160],pickable=True)]))

        # Diagnostics
        st.header("Corrélation et diagnostics")
        corr=df[['TEMP','MONTANT']].corr().loc['TEMP','MONTANT']; st.write(f"Corrélation météo-montant : {corr:.2f}")
        fig, ax=plt.subplots(); ax.scatter(df['TEMP'],df['MONTANT'],alpha=0.3)
        valid=df[['TEMP','MONTANT']].dropna();
        if len(valid)>1:
            coef=np.polyfit(valid['TEMP'],valid['MONTANT'],1);ax.plot(valid['TEMP'],coef[0]*valid['TEMP']+coef[1],color='red')
        ax.set(xlabel='Température',ylabel='Montant');st.pyplot(fig)
        bins=[-np.inf,5,15,np.inf];labels=['<5','5-15','>15'];df['bin']=pd.cut(df['TEMP'],bins=bins,labels=labels)
        st.subheader("Panier moyen par classes météo");st.bar_chart(df.groupby('bin')['MONTANT'].mean())
        if len(valid)>1:
            lr=LinearRegression().fit(valid[['TEMP']],valid['MONTANT']);st.write(f"Sensibilité : {lr.coef_[0]:.2f} €/°C")
        q1,q3=df['MONTANT'].quantile([0.25,0.75]);iqr=q3-q1;out=df[(df['MONTANT']<q1-1.5*iqr)|(df['MONTANT']>q3+1.5*iqr)]
        st.subheader("Outliers");st.write(len(out));st.write(out['TYPE_COMMERCE'].value_counts())

        # Segmentation
        st.header("Segmentation clients")
        feats=df[['MONTANT','HOUR','TEMP']].dropna();
        if len(feats)>=3:
            X=StandardScaler().fit_transform(feats)
            km=KMeans(3,random_state=42).fit(X);df['km']=km.labels_
            st.subheader("KMeans");st.bar_chart(df['km'].value_counts())
            db=DBSCAN(eps=0.5,min_samples=5).fit(X);df['db']=db.labels_
            st.subheader("DBSCAN");st.bar_chart(df['db'].value_counts())
            prof=df.groupby('km').agg({'MONTANT':'mean','HOUR':'mean','TEMP':'mean','REF_MARCHAND':'count'}).rename(columns={'REF_MARCHAND':'nb_tx'})
            st.dataframe(prof)
        else:
            st.write("⚠️ Pas assez de données pour la segmentation.")

        st.info("Sections prédictives (forecasting, alerting) à venir.")

    # === ÉTAPE 2 – AGENT CONVERSATIONAL EN DEHORS DU IF ===
    st.header("💬 Interrogez l'agent BI")
    if "chat_history" not in st.session_state: st.session_state.chat_history=[]
    if "summary_bi" not in st.session_state:
        resume=f"TX total: {len(df)}\nCA total: {df['MONTANT'].sum():.2f}€"
        st.session_state.summary_bi=resume
    user_input=st.text_input("Votre question :",key="chat_input")
    if st.button("Envoyer",key="send_btn") and user_input:
        messages=[{"role":"system","content":"Tu es un assistant BI expert."},
                  {"role":"user","content":f"Résumé :\n{st.session_state.summary_bi}"}]+st.session_state.chat_history[-5:]+[{"role":"user","content":user_input}]
        with st.spinner("L'agent réfléchit..."):
            import openai
            resp=openai.chat.completions.create(model="gpt-4",messages=messages)
            reply=resp.choices[0].message.content
            st.session_state.chat_history.append({"role":"user","content":user_input})
            st.session_state.chat_history.append({"role":"assistant","content":reply})
        if "```python" in reply:
            import re
            m=re.search(r"```python\n(.*?)```",reply,re.DOTALL)
            if m:
                code=m.group(1);st.code(code,language="python")
                try: exec(code,{}, {"df":df})
                except Exception as e: st.error(f"Erreur code: {e}")
            else: st.markdown(reply)
        else: st.markdown(reply)
    for msg in st.session_state.chat_history:
        icon="👤" if msg['role']=='user' else "🤖"
        st.markdown(f"**{icon} {msg['content']}**")
else:
    st.warning("Veuillez charger les 3 fichiers Excel pour continuer.")
