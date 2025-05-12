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
from pptx import Presentation
from pptx.util import Inches
import tempfile

st.set_page_config(layout="wide")
st.title("Smile Agent – Rapport BI & Agent Conversationnel")

# --- 1. Upload des 3 fichiers + modèle PPTX ---
st.sidebar.header("Chargement des fichiers")
file_tx = st.sidebar.file_uploader("Transactions (Test)", ["csv","xlsx"], key="tx")
file_merch = st.sidebar.file_uploader("Caractéristiques marchands", ["csv","xlsx"], key="merch")
file_weather = st.sidebar.file_uploader("Données météo", ["csv","xlsx"], key="weather")
file_template = st.sidebar.file_uploader("Modèle PPTX (exemple.pptx)", type=["pptx"], key="template")

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

if df_tx is not None and df_merch is not None and df_weather is not None:
    st.success("✅ Tous les fichiers chargés")
    # --- 2. Préparation et fusion ---
    # (Nettoyage, datetime, fusion marchands & météo)
    df_tx['MONTANT'] = df_tx['MONTANT'].astype(str).str.replace(r"[^0-9,.-]", '', regex=True).str.replace(',', '.').astype(float)
    df_tx['DATETIME'] = pd.to_datetime(df_tx['DATE'].astype(str) + ' ' + df_tx['HEURE'].astype(str), dayfirst=True, errors='coerce')
    df_tx['HOUR'] = df_tx['DATETIME'].dt.hour
    df_tx['DAY'] = df_tx['DATETIME'].dt.date
    df_tx['WEEK'] = df_tx['DATETIME'].dt.to_period('W').apply(lambda r: r.start_time)
    df_tx['MONTH'] = df_tx['DATETIME'].dt.to_period('M').apply(lambda r: r.start_time)
    df_weather.columns = df_weather.columns.str.strip().str.replace(r"\s+", ' ', regex=True).str.upper()
    if 'TEMPÉRATURE' in df_weather.columns: df_weather.rename(columns={'TEMPÉRATURE':'TEMP'}, inplace=True)
    if 'CODE_POSTAL' in df_weather.columns and 'CODE POSTAL' not in df_weather.columns: df_weather.rename(columns={'CODE_POSTAL':'CODE POSTAL'}, inplace=True)
    df = df_tx.merge(df_merch.rename(columns={'Organization_type':'TYPE_COMMERCE'})[['REF_MARCHAND','TYPE_COMMERCE']], on='REF_MARCHAND', how='left')
    df = df.merge(df_weather[['CODE POSTAL','TEMP']], on='CODE POSTAL', how='left')
    nomi = pgeocode.Nominatim('fr')
    unique_cp = df['CODE POSTAL'].astype(str).str.zfill(5).drop_duplicates().tolist()
    geo = nomi.query_postal_code(unique_cp)
    df_geo = geo[['postal_code','latitude','longitude']].dropna()
    df_geo['postal_code'] = df_geo['postal_code'].astype(str).str.zfill(5)
    df_geo.rename(columns={'postal_code':'CODE POSTAL'}, inplace=True)

    # --- 3. Rapport BI détaillé ---
    if st.button("📄 Smile Magic Report"):
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

         # Validation du template
        if file_template is None:
            st.error("Veuillez uploader le modèle PPTX dans la barre latérale pour générer la présentation.")
            st.stop()
        # Sauvegarde temporaire du modèle
        tmp_tpl = tempfile.NamedTemporaryFile(suffix='.pptx', delete=False)
        tmp_tpl.write(file_template.read())
        tmp_tpl.flush()
        prs = Presentation(tmp_tpl.name)

        # Mise à jour page de garde
        cover = prs.slides[0]
        # Titre
        cover.shapes.title.text = "Smile Magic Report"
        # Date (s'il y a un placeholder subtitle)
        try:
            cover.placeholders[1].text = pd.Timestamp.today().strftime("%d %B %Y")
        except Exception:
            pass

        # Sections à traiter et leurs figures
        sections = {
            "Indicateurs descriptifs": plt.gcf(),  # récupère figure active ou stockez vos fig
            "Temporalité fine": plt.gcf(),
            "Analyse spatiale": plt.gcf(),
            "Corrélations et diagnostics": plt.gcf(),
            "Segmentation client": plt.gcf()
        }

        for title, fig in sections.items():
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = title
            # Explication auto via OpenAI
            prompt = f"Explique en quelques phrases les résultats de la section '{title}' d'un rapport BI basé sur des transactions." 
            resp = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role":"system","content":"Tu es un expert BI."},
                    {"role":"user","content":prompt}
                ]
            )
            explanation = resp.choices[0].message.content

            #Nouvel ajout
            for title, fig in sections.items():
                slide = prs.slides.add_slide(prs.slide_layouts[1])
                slide.shapes.title.text = title
                # … génération de 'explanation' …
                try:
                    slide.placeholders[1].text = explanation
                except (KeyError, IndexError):
                    # fallback : TextBox
                    from pptx.util import Pt, Inches
                    textbox = slide.shapes.add_textbox(Inches(1), Inches(1.5), Inches(8), Inches(1.5))
                    tf = textbox.text_frame
                    tf.text = explanation
                    for paragraph in tf.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(12)
            
            # Ajout graphique
            if fig is not None:
                img_tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                fig.savefig(img_tmp.name, bbox_inches='tight')
                slide.shapes.add_picture(img_tmp.name, Inches(1), Inches(2), width=Inches(8))

        output = 'rapport_bi.pptx'
        prs.save(output)
        st.success(f"🎉 Présentation générée : {output}")

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
    st.header("💬 Interrogez Smile Agent")
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
