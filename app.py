import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import os
import io
import csv

# Clé OpenAI depuis les secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Mémoire conversationnelle
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Smile & Pay – Agent IA d'analyse de données")

# Upload du fichier
uploaded_file = st.file_uploader("📁 Charge un fichier CSV ou XLSX", type=["csv", "xlsx"])

df = None  # Initialisation

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        raw = uploaded_file.read()
        # 1) Détection d'encodage
        encoding_used = None
        for enc in ("utf-8", "ISO-8859-1"):
            try:
                _ = raw.decode(enc)
                encoding_used = enc
                break
            except Exception:
                continue
        if encoding_used is None:
            st.error("❌ Impossible de détecter l'encodage du CSV.")
        else:
            # 2) Détection de délimiteur
            try:
                sample = raw.decode(encoding_used)[:2048]
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
                delimiter = dialect.delimiter
            except Exception:
                delimiter = ","
            # 3) Lecture initiale
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=delimiter,
                    encoding=encoding_used,
                    engine="python",
                )
                st.info(f"Fichier lu avec encodage `{encoding_used}` et délimiteur `{delimiter}`")
            except Exception:
                # 4) Fallback quoting=None pour gérer les guillemets mal formés
                try:
                    df = pd.read_csv(
                        io.BytesIO(raw),
                        sep=delimiter,
                        encoding=encoding_used,
                        engine="python",
                        quoting=csv.QUOTE_NONE,
                        on_bad_lines="warn",
                    )
                    st.warning(
                        "⚠️ Lecture réussie en désactivant la gestion des guillemets (quoting)."
                    )
                except Exception as e:
                    st.error(f"❌ Impossible de lire le CSV : {str(e)}")
    elif uploaded_file.name.endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded_file)
            st.info("📄 Fichier Excel lu avec succès.")
        except Exception as e:
            st.error(f"❌ Erreur de lecture du fichier Excel : {str(e)}")
    else:
        st.error("❌ Format non supporté. Merci de charger un fichier .csv ou .xlsx")

if df is not None:
    st.success("✅ Données chargées")
    st.dataframe(df.head())

    # Rapport automatique
    if st.button("📄 Générer un rapport automatique"):
        rapport = f"""
Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}

Types :
{df.dtypes.to_string()}

Valeurs manquantes :
{df.isnull().sum()[df.isnull().sum() > 0].to_string() if df.isnull().sum().any() else "Aucune"}

Statistiques :
{df.describe(include='all').transpose().to_string()}

Corrélations :
{df.corr(numeric_only=True).to_string()}
"""
        st.text_area("📊 Rapport", rapport, height=300)

    # Interaction IA
    st.subheader("💬 Pose une question à l'IA")
    user_input = st.text_area("Ex : Analyse les ventes par semaine")

    if st.button("Analyser avec l'IA") and user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        context = [
            {
                "role": "system",
                "content": "Tu es un analyste de données professionnel "
                           "avec des compétences en data engineering, data science et data analytics.",
            }
        ]
        context.extend(st.session_state.history[-5:])
        context.append(
            {"role": "user", "content": f"Données :\n{df.head(5).to_csv(index=False)}"}
        )

        with st.spinner("🧠 L'IA analyse..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=context,
            )
        reply = response.choices[0].message.content
        st.session_state.history.append({"role": "assistant", "content": reply})
        st.markdown(reply)

    # Graphiques
    st.subheader("📈 Graphique personnalisé")
    col_x = st.selectbox("Axe X", df.columns)
    col_y = st.selectbox("Axe Y", df.columns)

    if st.button("Afficher le graphique"):
        fig, ax = plt.subplots()
        df.plot(x=col_x, y=col_y, kind="bar", ax=ax)
        st.pyplot(fig)


