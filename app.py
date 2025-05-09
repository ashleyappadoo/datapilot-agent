import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io
import csv

# --- 1. Setup OpenAI ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- 2. Mémoire conversationnelle ---
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Smile & Pay – Agent IA d'analyse de données")

# --- 3. Fonction de diagnostic/plot horaire ---
def handle_hourly_plot_request(df, query):
    if "graph" in query and "tranche horaire" in query:
        # Colonnes obligatoires
        needed = {"HEURE", "MONTANT"}
        missing = needed - set(df.columns)

        if missing:
            st.error(f"❌ Colonnes manquantes pour ce graphique : {', '.join(missing)}.")
            return True  # on arrête, sans appel IA
        # toutes les colonnes sont présentes
        df = df.copy()
        df["HOUR"] = pd.to_datetime(df["HEURE"], errors="coerce").dt.hour
        # nombre de jours uniques
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
        ax.set_xlabel("Heure")
        ax.set_ylabel(f"Moyenne sur {days} jour(s)")
        st.pyplot(fig)
        return True

    return False

# --- 4. Upload & lecture du fichier ---
uploaded_file = st.file_uploader("📁 Charge un fichier CSV ou XLSX", type=["csv", "xlsx"])
df = None

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        raw = uploaded_file.read()
        # Détection encodage
        encoding_used = None
        for enc in ("utf-8", "ISO-8859-1"):
            try:
                raw.decode(enc)
                encoding_used = enc
                break
            except:
                pass
        if not encoding_used:
            st.error("❌ Impossible de détecter l'encodage CSV.")
        else:
            # Détection délimiteur
            try:
                sample = raw.decode(encoding_used)[:2048]
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
                delimiter = dialect.delimiter
            except:
                delimiter = ","
            # Lecture tolérante
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=delimiter, encoding=encoding_used, engine="python")
                st.info(f"Lu avec encodage={encoding_used}, délimiteur='{delimiter}'")
            except:
                try:
                    df = pd.read_csv(io.BytesIO(raw),
                                     sep=delimiter,
                                     encoding=encoding_used,
                                     engine="python",
                                     quoting=csv.QUOTE_NONE,
                                     on_bad_lines="warn")
                    st.warning("Lecture réussie en désactivant les guillemets.")
                except Exception as e:
                    st.error(f"❌ Impossible de lire le CSV : {e}")
    else:  # Excel
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.info("Fichier Excel lu avec succès.")
        except Exception as e:
            st.error(f"❌ Erreur lecture Excel : {e}")

# --- 5. Si lecture OK, on poursuit ---
if df is not None:
    st.success("✅ Données chargées")
    st.dataframe(df.head())

    # Nettoyage MONTANT
    if "MONTANT" in df.columns:
        df["MONTANT"] = (
            df["MONTANT"]
            .astype(str)
            .str.replace(r"[^\d,.\-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df["MONTANT"] = pd.to_numeric(df["MONTANT"], errors="coerce")

    # Construction DATETIME
    if "DATE" in df.columns and "HEURE" in df.columns:
        df["DATETIME"] = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["HEURE"].astype(str),
            dayfirst=True, errors="coerce"
        )

    # --- 5a. Rapport BI détaillé ---
    if st.button("📄 Générer un rapport BI détaillé"):
        with st.spinner("📊 Génération du rapport…"):
            total_tx = len(df)
            total_amount = df["MONTANT"].sum() if "MONTANT" in df.columns else 0.0
            avg_amount = df["MONTANT"].mean() if "MONTANT" in df.columns else 0.0

            # Série temporelle si possible
            if "DATETIME" in df.columns and df["DATETIME"].notna().any():
                ts = df.set_index("DATETIME").resample("D")["MONTANT"].agg(nb_tx="count", volume="sum")
                fig1, ax1 = plt.subplots()
                ts["nb_tx"].plot(ax=ax1); ax1.set_title("Tx par jour"); st.pyplot(fig1)
                fig2, ax2 = plt.subplots()
                ts["volume"].plot(ax=ax2); ax2.set_title("Volume (€) par jour"); st.pyplot(fig2)

            # Distribution montants
            if "MONTANT" in df.columns:
                montants = df["MONTANT"].dropna().tolist()
                fig3, ax3 = plt.subplots()
                ax3.hist(montants, bins=30)
                ax3.set_title("Distribution des montants"); st.pyplot(fig3)

            # Top 5 marchands
            if "MARCHAND" in df.columns:
                top5 = df["MARCHAND"].value_counts().head(5)
                fig4, ax4 = plt.subplots()
                top5.plot(kind="bar", ax=ax4); ax4.set_title("Top 5 marchands"); st.pyplot(fig4)

            # Prompt BI
            prompt = f"""
Dataset: {total_tx} transactions, total={total_amount:.2f} €, moyen={avg_amount:.2f} €.
Colonnes: {', '.join(df.columns)}.
1) Résumé tendances.
2) Pics/creux.
3) Insights.
"""
            resp = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Tu es un expert BI."},
                    {"role": "user", "content": prompt},
                ],
            )
            st.markdown("### 📑 Rapport BI généré par l'IA")
            st.markdown(resp.choices[0].message.content)

    # --- 5b. Interaction libre et graphiques spécifiques ---
    st.subheader("💬 Pose une question ou demande un graphique")
    user_input = st.text_area("Ex : Génère un graphique nombre moyen de TX par tranche horaire")

    if st.button("Analyser"):
        query = user_input.lower()

        # 1) Cas graphique horaire
        if handle_hourly_plot_request(df, query):
            st.stop()

        # 2) Sinon appel OpenAI
        st.session_state.history.append({"role": "user", "content": user_input})
        context = [{"role": "system", "content": "Tu es un data analyst pro."}]
        context += st.session_state.history[-5:]
        context.append({"role": "user", "content": f"Données (extrait):\n{df.head(5).to_csv(index=False)}"})

        with st.spinner("🧠 L'IA réfléchit..."):
            resp = openai.chat.completions.create(
                model="gpt-4",
                messages=context,
            )
        answer = resp.choices[0].message.content
        st.session_state.history.append({"role": "assistant", "content": answer})
        st.markdown(answer)
