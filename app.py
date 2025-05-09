import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io, csv

st.set_page_config(layout="wide")
st.title("Smile & Pay – Rapport BI Transactions")

# 1. Upload
uploaded_file = st.file_uploader("📁 Charge un fichier CSV ou XLSX", type=["csv", "xlsx"])
if not uploaded_file:
    st.stop()

# 2. Lecture CSV/Excel
if uploaded_file.name.endswith(".csv"):
    raw = uploaded_file.read()
    # sniff encoding & delimiter (utf-8/ISO & , ; \t)
    for enc in ("utf-8","ISO-8859-1"):
        try:
            text = raw.decode(enc)
            encoding = enc
            break
        except:
            encoding = None
    if encoding is None:
        st.error("Impossible de détecter l'encodage CSV.")
        st.stop()
    try:
        delim = csv.Sniffer().sniff(text[:2048], delimiters=[",",";","\t"]).delimiter
    except:
        delim = ","
    df = pd.read_csv(io.BytesIO(raw), sep=delim, encoding=encoding, engine="python",
                     quoting=csv.QUOTE_NONE, on_bad_lines="warn")
else:
    df = pd.read_excel(uploaded_file, engine="openpyxl")

# 3. Nettoyage montant
if "MONTANT" in df.columns:
    df["MONTANT"] = (
        df["MONTANT"].astype(str)
        .str.replace(r"[^\d,.\-]", "", regex=True)
        .str.replace(",", ".", regex=False)
    ).pipe(pd.to_numeric, errors="coerce")

# 4. Construction DATETIME & HOUR
if {"DATE","HEURE"}.issubset(df.columns):
    df["DATETIME"] = pd.to_datetime(
        df["DATE"].astype(str)+" "+df["HEURE"].astype(str),
        dayfirst=True, errors="coerce"
    )
    df["HOUR"] = df["DATETIME"].dt.hour
else:
    df["HOUR"] = pd.to_datetime(df["HEURE"], errors="coerce").dt.hour if "HEURE" in df.columns else pd.NA

# 5. Génération du rapport
if st.button("📄 Générer un rapport BI détaillé"):
    st.markdown("## 🗂️ Aperçu du dataset")
    st.write(f"- **Transactions :** {len(df):,}")
    st.write(f"- **Colonnes :** {len(df.columns)} → {', '.join(df.columns)}")

    # 5.1 Volume horaire
    st.markdown("### 1. Volume horaire des transactions")
    valid_h = df.dropna(subset=["HOUR"])
    if not valid_h.empty:
        # nombre de jours uniques
        if "DATETIME" in df.columns:
            days = df["DATETIME"].dt.date.nunique()
        elif "DATE" in df.columns:
            days = pd.to_datetime(df["DATE"],dayfirst=True,errors="coerce").dt.date.nunique()
        else:
            days = 1
        hourly_counts = valid_h.groupby("HOUR").size()
        hourly_avg = (hourly_counts / days).reindex(range(24), fill_value=0)
        # résumé
        peak_h = int(hourly_avg.idxmax()); peak_v = int(hourly_avg.max())
        low_h  = int(hourly_avg.idxmin()); low_v  = int(hourly_avg.min())
        st.write(f"- Moyenne de **{hourly_avg.mean():.2f}** TX/heure sur {days} jour(s)")
        st.write(f"- **Pic** à {peak_h}h → {peak_v:.0f} TX ; **Creux** à {low_h}h → {low_v:.0f} TX")
        # graphique
        fig, ax = plt.subplots(figsize=(6,3))
        hourly_avg.plot(kind="bar", ax=ax)
        ax.set_xlabel("Heure"); ax.set_ylabel("TX/heure (moy.)")
        st.pyplot(fig)
    else:
        st.write("Aucune donnée horaire disponible.")

    # 5.2 High-Value Transactions
    st.markdown("### 2. High-Value Transactions (90ᵉ centile)")
    if "MONTANT" in df.columns:
        p90 = df["MONTANT"].quantile(0.9)
        hv = df[df["MONTANT"] >= p90]
        st.write(f"- **90ᵉ centile** = {p90:.2f} € → {len(hv):,} TX")
        if "TYPE_DE_CARTE" in hv.columns:
            dist = hv["TYPE_DE_CARTE"].value_counts()
            st.write(dist.to_frame("Count"))
            fig, ax = plt.subplots(figsize=(4,3))
            dist.plot(kind="bar", ax=ax)
            ax.set_xlabel("Type de carte"); ax.set_ylabel("Nb TX")
            st.pyplot(fig)
    else:
        st.write("Colonne `MONTANT` manquante.")

    # 5.3 Outliers par carte (IQR)
    st.markdown("### 3. Transaction Outliers par type de carte")
    if "MONTANT" in df.columns:
        q1,q3 = df["MONTANT"].quantile([0.25,0.75])
        iqr = q3-q1
        low,high = q1-1.5*iqr, q3+1.5*iqr
        out = df[(df["MONTANT"]<low)|(df["MONTANT"]>high)]
        st.write(f"- **Outliers** détectés : {len(out):,} TX ({len(out)/len(df)*100:.2f} %)")
        if "TYPE_DE_CARTE" in df.columns:
            out_by = out["TYPE_DE_CARTE"].value_counts()
            st.write(out_by.to_frame("Outliers"))
    else:
        st.write("Impossible sans `MONTANT`.")

    # 5.4 Top 5 marchands
    st.markdown("### 4. Top 5 Marchands par volume")
    if "MARCHAND" in df.columns:
        top5 = df["MARCHAND"].value_counts().head(5)
        st.write(top5.to_frame("Nb TX"))
        fig, ax = plt.subplots(figsize=(5,3))
        top5.plot(kind="bar", ax=ax)
        ax.set_xlabel("Marchand"); ax.set_ylabel("Nb TX")
        st.pyplot(fig)
    else:
        st.write("Colonne `MARCHAND` manquante.")

    # 5.5 Montants par mode de paiement
    st.markdown("### 5. Montants par mode de paiement")
    if {"MODE","MONTANT"}.issubset(df.columns):
        stats = df.groupby("MODE")["MONTANT"].agg(["mean","std"]).round(2)
        st.write(stats)
        fig, ax = plt.subplots(figsize=(5,3))
        stats["mean"].plot(kind="bar", yerr=stats["std"], ax=ax, capsize=4)
        ax.set_xlabel("Mode"); ax.set_ylabel("Montant (€)")
        st.pyplot(fig)
    else:
        st.write("Colonnes `MODE` ou `MONTANT` manquantes.")

    # 5.6 Volume par code postal
    st.markdown("### 6. Volume par code postal")
    if "CODE POSTAL" in df.columns:
        grp = df["CODE POSTAL"].value_counts()
        st.write(f"- **Codes postaux uniques** : {grp.size:,}")
        st.write(f"- **Moyenne TX par code** : {grp.mean():.2f}")
        fig, ax = plt.subplots(figsize=(6,3))
        grp.head(20).plot(kind="bar", ax=ax)
        ax.set_xlabel("Code postal"); ax.set_ylabel("Nb TX")
        st.pyplot(fig)
    else:
        st.write("Colonne `CODE POSTAL` manquante.")
