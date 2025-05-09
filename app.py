import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io, csv
import openai

st.set_page_config(layout="wide")
st.title("Smile & Pay – Rapport BI & Agent Conversationnel")

# --- 1. Upload et lecture du fichier ---
uploaded_file = st.file_uploader("📁 Charge un fichier CSV ou XLSX", type=["csv", "xlsx"])
df = None

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        raw = uploaded_file.read()
        # Détection d'encodage
        encoding = None
        for enc in ("utf-8", "ISO-8859-1"):
            try:
                raw.decode(enc)
                encoding = enc
                break
            except:
                pass
        if not encoding:
            st.error("❌ Impossible de détecter l'encodage CSV.")
        else:
            # Détection du délimiteur
            try:
                sample = raw.decode(encoding)[:2048]
                delimiter = csv.Sniffer().sniff(sample, delimiters=[",",";","\t"]).delimiter
            except:
                delimiter = ","
            # Lecture tolérante
            try:
                df = pd.read_csv(io.BytesIO(raw),
                                 sep=delimiter,
                                 encoding=encoding,
                                 engine="python")
                st.info(f"Lu avec encodage={encoding}, délimiteur='{delimiter}'")
            except:
                try:
                    df = pd.read_csv(io.BytesIO(raw),
                                     sep=delimiter,
                                     encoding=encoding,
                                     engine="python",
                                     quoting=csv.QUOTE_NONE,
                                     on_bad_lines="warn")
                    st.warning("Lecture réussie en désactivant les guillemets.")
                except Exception as e:
                    st.error(f"❌ Impossible de lire le CSV : {e}")
    else:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.info("Fichier Excel lu avec succès.")
        except Exception as e:
            st.error(f"❌ Erreur lecture Excel : {e}")

# --- 2. Préparation des données ---
if df is not None:
    st.success("✅ Données chargées")
    # Nettoyage MONTANT en float
    if "MONTANT" in df.columns:
        df["MONTANT"] = (
            df["MONTANT"]
            .astype(str)
            .str.replace(r"[^\d,.\-]", "", regex=True)
            .str.replace(",", ".", regex=False)
        ).pipe(pd.to_numeric, errors="coerce")
    # Construction DATETIME et HOUR
    if {"DATE","HEURE"}.issubset(df.columns):
        df["DATETIME"] = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["HEURE"].astype(str),
            dayfirst=True,
            errors="coerce"
        )
        df["HOUR"] = df["DATETIME"].dt.hour
    elif "HEURE" in df.columns:
        df["HOUR"] = pd.to_datetime(df["HEURE"], errors="coerce").dt.hour

    # --- 3. Génération du rapport BI détaillé ---
    if st.button("📄 Générer un rapport BI détaillé"):
        st.markdown("## 🗂️ Aperçu du dataset")
        st.write(f"- **Transactions :** {len(df):,}")
        st.write(f"- **Colonnes :** {len(df.columns)} → {', '.join(df.columns)}")

        # Volume horaire
        st.markdown("### 1. Volume horaire des transactions")
        if "HOUR" in df.columns:
            days = df["DATETIME"].dt.date.nunique() if "DATETIME" in df.columns else 1
            hourly_avg = (df.groupby("HOUR").size() / days).reindex(range(24), fill_value=0)
            peak, low = hourly_avg.idxmax(), hourly_avg.idxmin()
            st.write(f"- Moyenne de **{hourly_avg.mean():.2f}** TX/heure sur {days} jour(s)")
            st.write(f"- Pic à {peak}h ({int(hourly_avg.max())} TX), creux à {low}h ({int(hourly_avg.min())} TX)")
            fig, ax = plt.subplots(figsize=(6,3))
            hourly_avg.plot.bar(ax=ax)
            ax.set_xlabel("Heure"); ax.set_ylabel("TX/heure (moy.)")
            st.pyplot(fig)
        else:
            st.write("Colonne `HOUR` manquante — impossible de tracer le volume horaire.")

        # High-Value Transactions (90ᵉ centile)
        st.markdown("### 2. High-Value Transactions (90ᵉ centile)")
        if "MONTANT" in df.columns:
            p90 = df["MONTANT"].quantile(0.9)
            hv = df[df["MONTANT"] >= p90]
            st.write(f"- **90ᵉ centile** = {p90:.2f} € → {len(hv):,} TX")
            if "TYPE_DE_CARTE" in hv.columns:
                dist = hv["TYPE_DE_CARTE"].value_counts()
                fig, ax = plt.subplots(figsize=(4,3))
                dist.plot(kind="bar", ax=ax)
                ax.set_xlabel("Type de carte"); ax.set_ylabel("Nb TX")
                st.pyplot(fig)
        else:
            st.write("Colonne `MONTANT` manquante.")

        # Top 5 marchands
        st.markdown("### 3. Top 5 Marchands par volume")
        if "MARCHAND" in df.columns:
            top5 = df["MARCHAND"].value_counts().head(5)
            st.write(top5.to_frame("Nb TX"))
            fig, ax = plt.subplots(figsize=(5,3))
            top5.plot(kind="bar", ax=ax)
            ax.set_xlabel("Marchand"); ax.set_ylabel("Nb TX")
            st.pyplot(fig)
        else:
            st.write("Colonne `MARCHAND` manquante.")

    # --- 4. Agent conversationnel ---
    st.markdown("---")
    st.header("💬 Agent Conversationnel")

    # Initialise l’historique
    if "history" not in st.session_state:
        st.session_state.history = []

    # Affiche l’historique
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**Vous :** {msg['content']}")
        else:
            st.markdown(f"**Agent :** {msg['content']}")

    # Utilise un formulaire pour le chat
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Votre requête :", "")
        submitted = st.form_submit_button("Envoyer")

        if submitted and user_input:
            query = user_input.strip()
            st.session_state.history.append({"role": "user", "content": query})

            # Intention : graphique horaire
            if "graph" in query.lower() and "tranche horaire" in query.lower():
                if {"HOUR","MONTANT"}.issubset(df.columns):
                    days = df["DATETIME"].dt.date.nunique() if "DATETIME" in df.columns else 1
                    counts = df.groupby("HOUR").size()
                    hourly = (counts / days).reindex(range(24), fill_value=0)
                    fig, ax = plt.subplots()
                    hourly.plot.bar(ax=ax)
                    ax.set_title("Moyenne des transactions par tranche horaire")
                    ax.set_xlabel("Heure"); ax.set_ylabel(f"TX moy. sur {days}j")
                    st.pyplot(fig)
                    response = "Voici votre graphique."
                else:
                    response = "❌ Colonnes `HEURE` ou `MONTANT` manquantes. Veuillez fournir ces données."

            else:
                # Appel à OpenAI pour tout le reste
                preview = df.head(5).to_csv(index=False)
                messages = [{"role": "system", "content": "Tu es un expert data analyste et BI."}]
                messages += st.session_state.history[-5:]
                messages.append({"role": "user", "content": f"Données (extrait) :\n{preview}\nQuestion : {query}"})

                with st.spinner("🧠 L'agent réfléchit..."):
                    resp = openai.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                    )
                response = resp.choices[0].message.content

            # Affiche et mémorise la réponse
            st.session_state.history.append({"role": "assistant", "content": response})
            # Après sortie du form, l'app va se recharger automatiquement

