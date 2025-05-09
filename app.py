import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io, csv
import openai

st.set_page_config(layout="wide")
st.title("Smile & Pay – Rapport BI Transactions & Agent Conversationnel")

# --- 1. Upload et lecture ---
uploaded_file = st.file_uploader("📁 Charge un fichier CSV ou XLSX", type=["csv", "xlsx"])
if not uploaded_file:
    st.stop()

# Lecture CSV / XLSX
if uploaded_file.name.endswith(".csv"):
    raw = uploaded_file.read()
    # détecte encodage
    for enc in ("utf-8", "ISO-8859-1"):
        try:
            text = raw.decode(enc)
            encoding = enc
            break
        except:
            encoding = None
    if not encoding:
        st.error("Impossible de détecter l'encodage CSV.")
        st.stop()
    try:
        delim = csv.Sniffer().sniff(text[:2048], delimiters=[",",";","\t"]).delimiter
    except:
        delim = ","
    df = pd.read_csv(io.BytesIO(raw), sep=delim, encoding=encoding,
                     engine="python", quoting=csv.QUOTE_NONE, on_bad_lines="warn")
else:
    df = pd.read_excel(uploaded_file, engine="openpyxl")

# Nettoyage montant en float
if "MONTANT" in df.columns:
    df["MONTANT"] = (
        df["MONTANT"].astype(str)
        .str.replace(r"[^\d,.\-]", "", regex=True)
        .str.replace(",", ".", regex=False)
    ).pipe(pd.to_numeric, errors="coerce")

# Construit DATETIME et HOUR
if {"DATE","HEURE"}.issubset(df.columns):
    df["DATETIME"] = pd.to_datetime(
        df["DATE"].astype(str) + " " + df["HEURE"].astype(str),
        dayfirst=True, errors="coerce"
    )
    df["HOUR"] = df["DATETIME"].dt.hour
elif "HEURE" in df.columns:
    df["HOUR"] = pd.to_datetime(df["HEURE"], errors="coerce").dt.hour

# --- 2. Génération du rapport BI détaillé ---
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
        st.write("Colonne `HOUR` manquante, impossible de tracer le volume horaire.")

    # (Autres sections du rapport comme précédemment...)
    # ...

# --- 3. Agent conversationnel ---
st.markdown("---")
st.header("💬 Agent Conversationnel")

# Initialisation mémoire
if "history" not in st.session_state:
    st.session_state.history = []

# Zone de saisie utilisateur
user_input = st.text_input("Votre requête :", "")

if st.button("Envoyer") and user_input:
    query = user_input.strip()
    st.session_state.history.append({"role": "user", "content": query})

    # Détection d'intention graphique horaire
    lowq = query.lower()
    if "graph" in lowq and "tranche horaire" in lowq:
        # vérifie colonnes
        if {"HOUR","MONTANT"}.issubset(df.columns):
            days = df["DATETIME"].dt.date.nunique() if "DATETIME" in df.columns else 1
            counts = df.groupby("HOUR").size()
            hourly = counts / days
            fig, ax = plt.subplots()
            hourly.plot.bar(ax=ax)
            ax.set_title("Moyenne des transactions par tranche horaire")
            ax.set_xlabel("Heure"); ax.set_ylabel(f"TX moy. sur {days}j")
            st.pyplot(fig)
            st.session_state.history.append({"role": "assistant", "content": "Voici votre graphique."})
        else:
            st.error("Il manque les colonnes `HEURE` ou `MONTANT`. Veuillez fournir ces données ou les créer d'abord.")
            st.session_state.history.append({"role": "assistant", "content": "Colonnes manquantes pour ce graphique."})
    else:
        # Pour toute autre question, on appelle OpenAI
        # Prépare le contexte : extrait des en-têtes et stats clés
        preview = df.head(5).to_csv(index=False)
        system = {"role": "system", "content": "Tu es un expert data analyste et BI."}
        user_msg = {
            "role": "user",
            "content": f"Données (extrait) :\n{preview}\nQuestion : {query}"
        }
        messages = [system] + st.session_state.history[-5:] + [user_msg]
        with st.spinner("🧠 Réflexion en cours..."):
            resp = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
            )
        answer = resp.choices[0].message.content
        st.markdown(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})

    # Réinitialise l’input pour la prochaine question
    st.experimental_rerun()
