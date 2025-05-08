import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io
from dotenv import load_dotenv
import os

# Chargement de la clé API
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Mémoire du contexte utilisateur
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Smile & Pay – Agent IA d'analyse de données")

# Upload du fichier
uploaded_file = st.file_uploader("📁 Charge un fichier CSV ou XLSX", type=["csv", "xlsx"])

if uploaded_file:
    # Lecture du fichier
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Données chargées avec succès !")
    st.dataframe(df.head())

    # Génération automatique d'un rapport de base
    if st.button("📄 Générer un rapport automatique"):
        rapport = f"""
Le fichier contient {df.shape[0]} lignes et {df.shape[1]} colonnes.

Types de données :
{df.dtypes.to_string()}

Colonnes avec valeurs manquantes :
{df.isnull().sum()[df.isnull().sum() > 0].to_string() if df.isnull().sum().any() else "Aucune"}

Statistiques descriptives :
{df.describe(include='all').transpose().to_string()}

Corrélations :
{df.corr(numeric_only=True).to_string()}
"""
        st.text_area("📊 Rapport automatique", rapport, height=400)

    # Entrée utilisateur pour interaction
    st.subheader("💬 Pose une question à l'IA")
    user_input = st.text_area("Exemple : Donne-moi une analyse des ventes par jour.")

    if st.button("Analyser avec l'IA") and user_input:
        # Ajout au contexte
        st.session_state.history.append({"role": "user", "content": user_input})

        # Construction du contexte complet
        context = [{"role": "system", "content": "Tu es un assistant expert en data analytics."}]
        for message in st.session_state.history[-5:]:  # garder les 5 derniers messages
            context.append(message)

        # Ajout d'un aperçu des données
        preview = df.head(10).to_csv(index=False)
        context.append({"role": "user", "content": f"Voici un extrait du fichier pour t'aider :\n{preview}"})

        # Appel à OpenAI
        with st.spinner("🧠 L'IA réfléchit..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=context
            )
        output = response.choices[0].message.content
        st.session_state.history.append({"role": "assistant", "content": output})
        st.markdown(output)

    # Graphiques simples
    st.subheader("📈 Crée un graphique personnalisé")
    col_x = st.selectbox("Axe X", df.columns)
    col_y = st.selectbox("Axe Y", df.columns)

    if st.button("Afficher le graphique"):
        fig, ax = plt.subplots()
        df.plot(x=col_x, y=col_y, kind="bar", ax=ax)
        st.pyplot(fig)
