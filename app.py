import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io
import csv
import os

def handle_hourly_plot_request(df, query):
    """
    Traite la requête de type 'tranche horaire'.
    Retourne True si on a généré le graphique (et donc on arrête),
    ou False pour passer à l'appel normal à l'IA.
    """
    # On détecte l’intention
    if "graph" in query and "tranche horaire" in query:
        needed = {"HEURE", "MONTANT"}
        missing = needed - set(df.columns)

        if missing:
            # Colonnes manquantes : on propose de les créer
            st.error(f"❌ Impossible de générer le graphique : colonnes manquantes : {', '.join(missing)}.")
            if st.button(f"➕ Créer automatiquement la colonne {missing.pop()}"):
                # Exemple : si manque HEURE, on propose comment la générer
                if "HEURE" not in df.columns and "DATETIME" in df.columns:
                    df["HEURE"] = df["DATETIME"].dt.hour
                elif "HEURE" not in df.columns and "DATE" in df.columns and "HEURE_STR" in df.columns:
                    # cas hypothétique
                    df["HEURE"] = pd.to_datetime(df["HEURE_STR"]).dt.hour
                st.experimental_rerun()
            return True  # on stoppe ici (on veut attendre le rerun)
        else:
            # Toutes les colonnes sont là, on trace :
            df["HOUR"] = pd.to_datetime(df["HEURE"], errors="coerce").dt.hour
            # Calcul du nombre de jours
            if "DATETIME" in df.columns:
                days = df["DATETIME"].dt.date.nunique()
            elif "DATE" in df.columns:
                days = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce").dt.date.nunique()
            else:
                days = 1
            counts = df.groupby("HOUR").size()
            hourly = counts / days

            fig, ax = plt.subplots()
            hourly.plot(kind="bar", ax=ax)
            ax.set_title("Nombre moyen de transactions par tranche horaire")
            ax.set_xlabel("Heure de la journée")
            ax.set_ylabel(f"Moyenne de TX sur {days} jour(s)")
            st.pyplot(fig)
            return True

    return False


# 1. Clé OpenAI depuis les secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 2. Mémoire conversationnelle
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Smile & Pay – Agent IA d'analyse de données")

# 3. Upload
uploaded_file = st.file_uploader("📁 Charge un fichier CSV ou XLSX", type=["csv", "xlsx"])
df = None

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        raw = uploaded_file.read()
        # 3.1 Détection d'encodage
        encoding_used = None
        for enc in ("utf-8", "ISO-8859-1"):
            try:
                _ = raw.decode(enc)
                encoding_used = enc
                break
            except:
                continue
        if encoding_used is None:
            st.error("❌ Impossible de détecter l'encodage du CSV.")
        else:
            # 3.2 Détection du délimiteur
            try:
                sample = raw.decode(encoding_used)[:2048]
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
                delimiter = dialect.delimiter
            except:
                delimiter = ","
            # 3.3 Lecture tolérante
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=delimiter,
                    encoding=encoding_used,
                    engine="python",
                )
                st.info(f"Fichier lu avec encodage `{encoding_used}` et délimiteur `{delimiter}`")
            except Exception:
                # 3.4 Fallback quoting=None
                try:
                    df = pd.read_csv(
                        io.BytesIO(raw),
                        sep=delimiter,
                        encoding=encoding_used,
                        engine="python",
                        quoting=csv.QUOTE_NONE,
                        on_bad_lines="warn",
                    )
                    st.warning("⚠️ Lecture réussie en désactivant la gestion des guillemets.")
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

# 4. Si DataFrame présent, suite
if df is not None:
    st.success("✅ Données chargées")
    st.dataframe(df.head())

    # 4.1 Nettoyage des montants
    if "MONTANT" in df.columns:
        df["MONTANT"] = (
            df["MONTANT"]
            .astype(str)
            .str.replace(r"[^\d,.\-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df["MONTANT"] = pd.to_numeric(df["MONTANT"], errors="coerce")
    if "MONTANT_INITIAL" in df.columns:
        df["MONTANT_INITIAL"] = (
            df["MONTANT_INITIAL"]
            .astype(str)
            .str.replace(r"[^\d,.\-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df["MONTANT_INITIAL"] = pd.to_numeric(df["MONTANT_INITIAL"], errors="coerce")

    # 5. Rapport BI détaillé
    if st.button("📄 Générer un rapport BI détaillé"):
        with st.spinner("📊 Calcul et génération des graphiques…"):
            # 5.1 Construction du datetime
            if "DATE" in df.columns and "HEURE" in df.columns:
                df["DATETIME"] = pd.to_datetime(
                    df["DATE"].astype(str) + " " + df["HEURE"].astype(str),
                    dayfirst=True, errors="coerce"
                )
            else:
                df["DATETIME"] = pd.NaT

            # 5.2 Indicateurs clé
            total_tx = len(df)
            total_amount = df["MONTANT"].sum() if "MONTANT" in df.columns else 0.0
            avg_amount = df["MONTANT"].mean() if "MONTANT" in df.columns else 0.0

            # 5.3 Série temporelle
            if df["DATETIME"].notna().any():
                ts = df.set_index("DATETIME").resample("D")["MONTANT"].agg(nb_tx="count", volume="sum")

                fig1, ax1 = plt.subplots()
                ts["nb_tx"].plot(ax=ax1)
                ax1.set_title("Nombre de transactions par jour")
                ax1.set_ylabel("Nombre de TX")
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                ts["volume"].plot(ax=ax2)
                ax2.set_title("Volume des transactions (€) par jour")
                ax2.set_ylabel("€")
                st.pyplot(fig2)

            # 5.4 Distribution des montants
            if "MONTANT" in df.columns:
                montants = df["MONTANT"].dropna().tolist()
                fig3, ax3 = plt.subplots()
                ax3.hist(montants, bins=30)
                ax3.set_title("Distribution des montants")
                ax3.set_xlabel("Montant (€)")
                ax3.set_ylabel("Fréquence")
                st.pyplot(fig3)

            # 5.5 Top 5 des marchands
            if "MARCHAND" in df.columns:
                top_merch = df["MARCHAND"].value_counts().head(5)
                fig4, ax4 = plt.subplots()
                top_merch.plot(kind="bar", ax=ax4)
                ax4.set_title("Top 5 des marchands (nb TX)")
                ax4.set_ylabel("Nombre de TX")
                st.pyplot(fig4)

            # 5.6 Prompt OpenAI pour rapport narratif
            prompt = f"""
Tu es un expert BI sur un système de paiement. Le dataset contient {total_tx} transactions,
pour un montant total de {total_amount:.2f} € et un montant moyen de {avg_amount:.2f} €.
Les colonnes principales : {', '.join(df.columns)}.

Rédige un rapport structuré en français :
1) Résumé des tendances quotidiennes
2) Points remarquables (pics, creux)
3) Recommandations / insights
Fais référence aux graphiques générés (transactions par jour, volumes, distribution, top marchands).
"""
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Tu es un analyste BI expert."},
                    {"role": "user", "content": prompt},
                ],
            )
            rapport_bi = response.choices[0].message.content

        st.markdown("### 📑 Rapport BI généré par l'IA")
        st.markdown(rapport_bi)

    # 6. Interaction libre avec reconnaissance des requêtes graphiques
st.subheader("💬 Pose une question à l'IA")
user_input = st.text_area("Ex : Quelles ont été les tendances du weekend ? Ou : Génère un graphique nombre moyen de TX par tranche horaire")

if st.button("Analyser avec l'IA") and user_input:
    query = user_input.lower()

    # 2.1. Diagnostic pour graphique horaire
    plotted = handle_hourly_plot_request(df, query)
    if plotted:
        st.stop()   # on arrête ici si on a géré la demande

    # 2.2. Sinon on laisse l'IA répondre
    st.session_state.history.append({"role": "user", "content": user_input})
    context = [{"role": "system", "content": "Tu es un analyste de données professionnel."}]
    context += st.session_state.history[-5:]
    context.append({"role": "user", "content": f"Données (extrait) :\n{df.head(5).to_csv(index=False)}"})

    with st.spinner("🧠 L'IA analyse…"):
        resp = openai.chat.completions.create(
            model="gpt-4",
            messages=context,
        )
    reply = resp.choices[0].message.content
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.markdown(reply)

    # 7. Graphique personnalisé
    st.subheader("📈 Graphique personnalisé")
    col_x = st.selectbox("Axe X", df.columns)
    col_y = st.selectbox("Axe Y", df.columns)
    if st.button("Afficher le graphique"):
        fig, ax = plt.subplots()
        df.plot(x=col_x, y=col_y, kind="bar", ax=ax)
        st.pyplot(fig)
