# -*- coding: utf-8 -*-
"""
Détection & prédiction d'attaques DDoS — application Streamlit
Basée sur l'atelier « Détection d'attaques DDoS avec Machine Learning ».

Lancer :  streamlit run app.py
Identifiants :  login = nada   |   mot de passe = nadanada

NB : le dataset est SYNTHÉTIQUE (usage pédagogique). Les scores obtenus ne
démontrent pas une efficacité sur du trafic réseau réel.
"""
import hmac
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "dataset.csv"
VALID_USER = "nada"
VALID_PASSWORD = "nadanada"

DROP_COLS = ["label", "timestamp", "src_ip", "dst_ip", "src_port"]
CAT_COLS = ["protocol", "dst_port"]
NUM_COLS = ["duration_ms", "packet_count", "byte_count", "packets_per_second",
            "bytes_per_second", "avg_packet_size_bytes", "syn_count",
            "ack_count", "connections_10s", "unique_sources_10s"]
SCALED_MODELS = ["Régression linéaire", "Régression logistique", "SVM"]
CLASS_NAMES = ["Normal", "DDoS"]

st.set_page_config(page_title="Détection DDoS", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------
# Authentification
# --------------------------------------------------------------------------
def check_login() -> bool:
    """Affiche le formulaire de connexion tant que l'utilisateur n'est pas connecté."""
    if st.session_state.get("auth_ok"):
        return True

    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        st.title("🛡️ Détection DDoS")
        st.caption("Veuillez vous connecter pour accéder à l'application.")
        with st.form("login_form"):
            user = st.text_input("Login")
            pwd = st.text_input("Mot de passe", type="password")
            ok = st.form_submit_button("Se connecter", use_container_width=True)
        if ok:
            good_user = hmac.compare_digest(user.encode(), VALID_USER.encode())
            good_pwd = hmac.compare_digest(pwd.encode(), VALID_PASSWORD.encode())
            if good_user and good_pwd:
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Login ou mot de passe incorrect.")
    return False


# --------------------------------------------------------------------------
# Données & prétraitement (même logique que l'atelier)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame):
    """Doublons supprimés, NaN remplacés par 0. Retourne (df_propre, nb_doublons, nb_nan)."""
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    n_nan = int(df.isna().sum().sum())
    df = df.fillna(0)
    return df, n_dup, n_nan


def build_features(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """X = trafic sans label / IP / timestamp / src_port + get_dummies(protocol, dst_port).
    Si `columns` est fourni, aligne X sur les colonnes vues à l'entraînement."""
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = X.fillna(0)
    X = pd.get_dummies(X, columns=CAT_COLS, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


@st.cache_resource(show_spinner="Entraînement des 5 modèles en cours…")
def train_all():
    raw = load_raw()
    df, n_dup, n_nan = clean(raw)

    y = df["label"]
    X = build_features(df)

    # 70 % train / 15 % validation / 15 % test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # Scaler ajusté uniquement sur train (pas de fuite de données)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    models = {
        "Régression linéaire": LinearRegression().fit(X_train_s, y_train),
        "Régression logistique": LogisticRegression(
            max_iter=1000, random_state=42).fit(X_train_s, y_train),
        "SVM": SVC(kernel="rbf", C=1.0).fit(X_train_s, y_train),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=2).fit(X_train, y_train),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=2).fit(X_train, y_train),
    }

    bundle = dict(models=models, scaler=scaler, columns=list(X.columns),
                  X_val=X_val, X_val_s=X_val_s, y_val=y_val,
                  X_test=X_test, X_test_s=X_test_s, y_test=y_test,
                  sizes=(len(X_train), len(X_val), len(X_test)),
                  n_dup=n_dup, n_nan=n_nan, n_clean=len(df))

    # Prédictions sur validation + tableau comparatif
    rows, val_pred = [], {}
    for nom in models:
        pred = predict_with(bundle, nom, X_val, X_val_s)
        val_pred[nom] = pred
        rows.append({
            "Modèle": nom,
            "Accuracy": accuracy_score(y_val, pred),
            "Précision": precision_score(y_val, pred, zero_division=0),
            "Recall": recall_score(y_val, pred, zero_division=0),
            "F1-score": f1_score(y_val, pred, zero_division=0),
        })
    table = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    bundle["val_pred"] = val_pred
    bundle["table"] = table
    bundle["best"] = table.iloc[0]["Modèle"]
    return bundle


def predict_with(bundle, nom, X_raw, X_scaled=None):
    """Prédit 0/1 avec le modèle `nom`. Fournir X_raw (non scalé) ; X_scaled optionnel."""
    model = bundle["models"][nom]
    if nom in SCALED_MODELS:
        Xs = X_scaled if X_scaled is not None else bundle["scaler"].transform(X_raw)
        out = model.predict(Xs)
    else:
        out = model.predict(X_raw)
    if nom == "Régression linéaire":
        out = (out >= 0.5).astype(int)
    return np.asarray(out).astype(int)


def attack_probability(bundle, nom, X_raw):
    """Probabilité d'attaque (None si le modèle n'en fournit pas)."""
    model = bundle["models"][nom]
    if nom == "Régression linéaire":
        return np.clip(model.predict(bundle["scaler"].transform(X_raw)), 0, 1)
    if nom == "SVM":
        return None
    Xin = bundle["scaler"].transform(X_raw) if nom in SCALED_MODELS else X_raw
    return model.predict_proba(Xin)[:, 1]


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_stats(bundle):
    st.header("📊 Données & statistiques")
    raw = load_raw()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{len(raw):,}".replace(",", " "))
    c2.metric("Colonnes", raw.shape[1])
    c3.metric("Doublons supprimés", bundle["n_dup"])
    c4.metric("Valeurs manquantes (→ 0)", bundle["n_nan"])

    st.info("Dataset **synthétique** (pédagogique) : il simule des logs agrégés de flux "
            "réseau. Les résultats ne se transfèrent pas automatiquement au trafic réel.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Répartition des classes")
        counts = raw["label"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        bars = ax.bar(CLASS_NAMES, counts.values, color=["#4C9F70", "#D9534F"])
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / len(raw):.1%})",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, counts.max() * 1.25)
        ax.set_ylabel("Nombre de flux")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Protocoles par classe")
        ct = pd.crosstab(raw["protocol"], raw["label"].map(dict(enumerate(CLASS_NAMES))))
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ct.plot.bar(ax=ax, color=["#D9534F", "#4C9F70"][:ct.shape[1]], rotation=0)
        ax.set_xlabel("")
        ax.set_ylabel("Nombre de flux")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Aperçu des données")
    st.dataframe(raw.head(20), use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(raw.describe().T.round(2), use_container_width=True)

    st.subheader("Valeurs manquantes par colonne")
    nan_df = raw.isna().sum().rename("NaN").to_frame()
    st.dataframe(nan_df[nan_df["NaN"] > 0].T, use_container_width=True)

    st.subheader("Moyenne des variables par classe")
    means = raw.groupby(raw["label"].map(dict(enumerate(CLASS_NAMES))))[NUM_COLS].mean().T
    st.dataframe(means.round(2), use_container_width=True)


def page_models(bundle):
    st.header("🤖 Modèles & métriques")
    n_tr, n_va, n_te = bundle["sizes"]
    st.caption(f"Découpage : train **{n_tr}** (70 %) · validation **{n_va}** (15 %) · "
               f"test **{n_te}** (15 %). Le meilleur modèle est choisi sur le F1-score de validation.")

    table = bundle["table"]
    st.subheader("Comparaison sur le jeu de validation")
    st.dataframe(
        table.style.format({c: "{:.3f}" for c in table.columns[1:]})
             .highlight_max(subset=table.columns[1:], color="#cfe8d5"),
        use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(table["Modèle"], table["F1-score"], color="#4C78A8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score (validation)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Détail d'un modèle")
    nom = st.selectbox("Modèle", list(bundle["models"]),
                       index=list(bundle["models"]).index(bundle["best"]))
    jeu = st.radio("Jeu d'évaluation", ["Validation", "Test final"], horizontal=True)
    if jeu == "Validation":
        y_true, y_pred = bundle["y_val"], bundle["val_pred"][nom]
    else:
        y_true = bundle["y_test"]
        y_pred = predict_with(bundle, nom, bundle["X_test"], bundle["X_test_s"])
        if nom != bundle["best"]:
            st.warning("Le jeu de test doit servir uniquement à vérifier le meilleur modèle "
                       "choisi sur la validation. Cette vue est donnée à titre indicatif.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
    c2.metric("Précision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
    c3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
    c4.metric("F1-score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=CLASS_NAMES, cmap="Blues", ax=ax)
        ax.set_title(f"Matrice de confusion — {jeu.lower()}")
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)
        st.caption("Faux négatifs = attaques non détectées · "
                   "Faux positifs = trafic normal signalé à tort.")
    if nom == "Régression linéaire":
        st.caption("⚠️ La régression linéaire est une baseline pédagogique : ce n'est pas un "
                   "vrai classifieur probabiliste (seuil à 0,5).")


def show_result(pred: int, proba):
    if pred == 1:
        st.error("🚨 **DDoS** — trafic suspect détecté")
    else:
        st.success("✅ **Normal** — trafic légitime")
    if proba is not None:
        st.progress(float(proba), text=f"Probabilité d'attaque : {float(proba):.1%}")


def page_predict(bundle):
    st.header("🔮 Prédiction : Normal / DDoS")
    raw = load_raw()
    nom = st.selectbox("Modèle utilisé", list(bundle["models"]),
                       index=list(bundle["models"]).index(bundle["best"]),
                       help="Par défaut : le meilleur modèle selon le F1-score de validation.")

    tab_manual, tab_sample, tab_batch = st.tabs(
        ["✍️ Saisie manuelle", "🎲 Exemple du dataset", "📁 Fichier CSV"])

    # ---- Saisie manuelle ----
    with tab_manual:
        st.caption("Renseignez les caractéristiques d'un flux réseau.")
        med = raw[NUM_COLS].median()
        with st.form("manual_form"):
            c1, c2 = st.columns(2)
            protocol = c1.selectbox("Protocole", sorted(raw["protocol"].unique()))
            dst_port = c2.selectbox("Port de destination", sorted(raw["dst_port"].unique()))
            vals = {}
            cols = st.columns(3)
            for i, col in enumerate(NUM_COLS):
                vals[col] = cols[i % 3].number_input(
                    col, min_value=0.0, value=float(round(med[col], 2)), format="%.2f")
            go = st.form_submit_button("Prédire", type="primary")
        if go:
            row = pd.DataFrame([{"protocol": protocol, "dst_port": dst_port, **vals}])
            X = build_features(row, bundle["columns"])
            pred = int(predict_with(bundle, nom, X)[0])
            proba = attack_probability(bundle, nom, X)
            show_result(pred, None if proba is None else proba[0])

    # ---- Exemple du dataset ----
    with tab_sample:
        st.caption("Tire un flux au hasard dans le dataset et compare la prédiction à la vraie étiquette.")
        if st.button("Tirer un flux aléatoire"):
            st.session_state["sample_idx"] = int(np.random.randint(len(raw)))
        idx = st.session_state.get("sample_idx")
        if idx is not None:
            row = raw.iloc[[idx]]
            st.dataframe(row, use_container_width=True, hide_index=True)
            X = build_features(row.fillna(0), bundle["columns"])
            pred = int(predict_with(bundle, nom, X)[0])
            proba = attack_probability(bundle, nom, X)
            show_result(pred, None if proba is None else proba[0])
            vrai = int(row["label"].iloc[0])
            st.write(f"Vraie étiquette : **{CLASS_NAMES[vrai]}** — "
                     + ("✔️ prédiction correcte" if vrai == pred else "❌ prédiction erronée"))

    # ---- Fichier CSV ----
    with tab_batch:
        st.caption("Chargez un CSV de flux (mêmes colonnes que le dataset, la colonne `label` est facultative).")
        up = st.file_uploader("Fichier CSV", type="csv")
        if up is not None:
            try:
                new = pd.read_csv(up)
                missing = [c for c in CAT_COLS + NUM_COLS if c not in new.columns]
                if missing:
                    st.error(f"Colonnes manquantes : {', '.join(missing)}")
                else:
                    X = build_features(new, bundle["columns"])
                    pred = predict_with(bundle, nom, X)
                    out = new.copy()
                    out["prediction"] = np.where(pred == 1, "DDoS", "Normal")
                    proba = attack_probability(bundle, nom, X)
                    if proba is not None:
                        out["proba_attaque"] = np.round(proba, 4)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Flux analysés", len(out))
                    c2.metric("DDoS détectés", int((pred == 1).sum()))
                    c3.metric("Normaux", int((pred == 0).sum()))
                    if "label" in new.columns:
                        y_true = new["label"].astype(int)
                        st.write(f"Accuracy sur ce fichier : **{accuracy_score(y_true, pred):.3f}** · "
                                 f"F1 : **{f1_score(y_true, pred, zero_division=0):.3f}**")
                    st.dataframe(out.head(200), use_container_width=True)
                    st.download_button("⬇️ Télécharger les prédictions (CSV)",
                                       out.to_csv(index=False).encode("utf-8"),
                                       "predictions_ddos.csv", "text/csv")
            except Exception as exc:  # fichier illisible, etc.
                st.error(f"Impossible de traiter le fichier : {exc}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if not check_login():
        return

    with st.sidebar:
        st.title("🛡️ Détection DDoS")
        st.caption(f"Connecté : **{VALID_USER}**")
        page = st.radio("Navigation", ["📊 Données & stats", "🤖 Modèles & métriques",
                                       "🔮 Prédiction"])
        if st.button("Se déconnecter", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if not DATA_PATH.exists():
        st.error(f"Fichier introuvable : {DATA_PATH.name}. Placez dataset.csv à côté de app.py.")
        return

    bundle = train_all()
    if page.startswith("📊"):
        page_stats(bundle)
    elif page.startswith("🤖"):
        page_models(bundle)
    else:
        page_predict(bundle)


main()
