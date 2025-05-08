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

    # Rapport BI détaillé
if st.button("📄 Générer un rapport BI détaillé"):
    with st.spinner("📊 Calcul des indicateurs et génération des graphiques…"):
        # 1. Préparation du datetime (si colonnes DATE et HEURE existantes)
        if "DATE" in df.columns and "HEURE" in df.columns:
            df["DATETIME"] = pd.to_datetime(df["DATE"].astype(str) + " " + df["HEURE"].astype(str),
                                            dayfirst=True, errors="coerce")
        else:
            df["DATETIME"] = pd.NaT

        # 2. Indicateurs clés
        total_tx = len(df)
        total_amount = df["MONTANT"].sum() if "MONTANT" in df.columns else None
        avg_amount = df["MONTANT"].mean() if "MONTANT" in df.columns else None

        # 3. Série temporelle journalière
        if df["DATETIME"].notna().any():
            ts = (df.set_index("DATETIME")
                    .resample("D")["MONTANT"]
                    .agg(["count","sum"])
                    .rename(columns={"count":"nb_tx","sum":"volume"}))
            fig1, ax1 = plt.subplots()
            ts["nb_tx"].plot(ax=ax1)
            ax1.set_title("Nombre de transactions par jour")
            ax1.set_ylabel("Nombre de TX")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ts["volume"].plot(ax=ax2)
            ax2.set_title("Volume des transactions (€) par jour")
            ax2.set_ylabel("Montant total")
            st.pyplot(fig2)

        # 4. Distribution des montants
        if "MONTANT" in df.columns:
            fig3, ax3 = plt.subplots()
            df["MONTANT"].hist(bins=30, ax=ax3)
            ax3.set_title("Distribution des montants de transaction")
            ax3.set_xlabel("Montant (€)")
            ax3.set_ylabel("Fréquence")
            st.pyplot(fig3)

        # 5. Top 5 marchands (si existants)
        if "MARCHAND" in df.columns:
            top_merch = df["MARCHAND"].value_counts().head(5)
            fig4, ax4 = plt.subplots()
            top_merch.plot(kind="bar", ax=ax4)
            ax4.set_title("Top 5 des marchands par nombre de TX")
            ax4.set_ylabel("Nombre de TX")
            st.pyplot(fig4)

        # 6. Préparation du prompt pour OpenAI
        prompt = f"""
Tu es un expert Business Intelligence pour un système de paiement.
Le dataset contient {total_tx} transactions.
Total du montant traité : {total_amount:.2f} €, montant moyen : {avg_amount:.2f} €.
Colonnes principales : {', '.join(df.columns.tolist())}.

Détaille pour moi :
- Un résumé des tendances (volume / nombre) observées.
- Les points remarquables (pics, creux).
- Des recommandations ou insights sur la base de ces données.
- Fais référence aux graphiques générés :
  1) Nombre de transactions par jour
  2) Volume des transactions par jour
  3) Distribution des montants
  4) Top 5 des marchands

Répond en français, sous forme de rapport structuré (titres, paragraphes).
"""

        # 7. Appel à OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Tu es un analyste BI expert."},
                {"role": "user", "content": prompt}
            ]
        )
        rapport_bi = response.choices[0].message.content

    # 8. Affichage du rapport
    st.markdown("### 📑 Rapport BI généré par l'IA")
    st.markdown(rapport_bi)


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


