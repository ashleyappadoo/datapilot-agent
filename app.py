import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import os

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
    try:
        # Vérification du type de fichier
        if uploaded_file.name.endswith(".csv"):
            try:
                # Tentative avec encodage utf-8 et séparateur virgule
                df = pd.read_csv(uploaded_file)
            except pd.errors.ParserError:
                try:
                    # Tentative avec séparateur point-virgule et encodage utf-8
                    df = pd.read_csv(uploaded_file, sep=";")
                except pd.errors.ParserError:
                    try:
                        # Tentative avec séparateur point-virgule et encodage ISO
                        df = pd.read_csv(uploaded_file, sep=";", encoding="ISO-8859-1")
                    except Exception as e:
                        st.error(f"❌ Impossible de lire le fichier CSV : {str(e)}")
                        df = None
            except UnicodeDecodeError:
                try:
                    # Tentative avec encodage ISO et séparateur virgule
                    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
                except Exception as e:
                    st.error(f"❌ Erreur d'encodage (ISO-8859-1) : {str(e)}")
                    df = None
        elif uploaded_file.name.endswith(".xlsx"):
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"❌ Erreur de lecture du fichier Excel : {str(e)}")
                df = None
        else:
            st.error("❌ Format non supporté. Merci de charger un fichier .csv ou .xlsx")
            df = None


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
        context = [{"role": "system", "content": "Tu es un analyste de données professionnel avec des compétences en data engineering, data scientist et data analytics."}]
        context.extend(st.session_state.history[-5:])
        context.append({"role": "user", "content": f"Données :\n{df.head(5).to_csv(index=False)}"})

        with st.spinner("🧠 L'IA analyse..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=context
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
