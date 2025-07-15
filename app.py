import streamlit as st
import pandas as pd
import plotly.express as px
import re
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import seaborn.objects as so
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from io import BytesIO
import joblib
from collections import Counter
from time import sleep

@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

@st.cache_resource
def load_models():
    return {
        "Régression Logistique": joblib.load("logistic_model.pkl"),
        "SVM": joblib.load("svm_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
    }

@st.cache_resource
def load_spacy():
    return spacy.load("fr_core_news_sm")

nlp = load_spacy()

vectorizer = load_vectorizer()
models = load_models()

labels = {0: "Négatif", 1: "Positif"}
colors = {"Positif": "#5CD677", "Négatif": "#E35757"}
emojis = {"Positif": "✅", "Négatif": "❌"}

st.set_page_config(page_title="Analyse de polarité", layout="centered")
st.title("Analyse de la polarité des avis")

tab_analyse, tab_evaluation, tab_interpretation = st.tabs(["🧠 Analyse des avis", "📊 Évaluation des modèles", "📝 Interprétation"])

def nettoyer_texte(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    doc = nlp(text)
    mots_utiles = [token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS and not token.is_punct and not token.is_space]
    return " ".join(mots_utiles)

def afficher_matrice_confusion(y_true, y_pred, nom):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=["Négatif", "Positif"], columns=["Prédit Nég", "Prédit Pos"])
    fig = px.imshow(df_cm, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    fig.update_layout(title_text=f"Matrice de confusion - {nom}")
    st.plotly_chart(fig, use_container_width=True)

def generer_nuage(texte, polar):
    wc = WordCloud(width=1000, height=600, background_color="white", max_words=100,
                   colormap="summer" if polar == "Positif" else "cool").generate(texte)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.subheader(f"Nuage de mots {polar} {emojis[polar]}")
    st.pyplot(fig)

# Onglet Analyse des avis
with tab_analyse:
    model_name = st.selectbox("Choisissez un modèle pour l'analyse", list(models.keys()))
    seuil = st.slider("🔍 Seuil de confiance minimum (%)", 0, 100, 90)

    # Colonnes pour la saisie
    col1, col2 = st.columns([1,1])
    with col1:
        avis_input = st.text_area("✍️ Entrez des avis (un par ligne) :", height=135)
    with col2:
        fichier = st.file_uploader("📄 Ou chargez un fichier CSV des avis", type=["csv"])
    
    # Boutons "Analyser" et "Rénitialiser"
    col_analyse, col_reset = st.columns([2, 2])
    with col_analyse:
        analyser_clicked = st.button("Analyser", type="primary")  # bouton bleu
    with col_reset:
        reinit_clicked = st.button("Réinitialiser")  # bouton gris

    # Action lors du clic sur le bouton Réinitialiser
    if reinit_clicked:
        st.session_state.clear()
        st.rerun()

    # Action lors du clic sur le bouton Analyser
    if analyser_clicked:
        avis_liste = []
        if fichier:
            try:
                df = pd.read_csv(fichier)
                if "review" not in df.columns:
                    st.error("❌ Le fichier doit contenir une colonne 'review'.")
                else:
                    avis_liste = df["review"].dropna().astype(str).tolist()
                    st.session_state["avis_csv"] = avis_liste
                    st.session_state["labels_csv"] = df["label"].tolist() if "label" in df.columns else None
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier : {e}")
        else:
            avis_liste = [a.strip() for a in avis_input.split("\n") if a.strip()]
        if not avis_liste:
             st.info("Veuillez saisir au moins un avis ou charger un fichier CSV.")
        else:
            with st.spinner("Nettoyage et analyse en cours..."):
                avis_nettoyes = []
                progress = st.progress(0)
                for i, a in enumerate(avis_liste):
                    avis_nettoyes.append(nettoyer_texte(a))
                    progress.progress((i + 1) / len(avis_liste))
                    sleep(0.01)
                vect = vectorizer.transform(avis_nettoyes)
                model = models[model_name]
                preds = model.predict(vect)
                try:
                    probas = model.predict_proba(vect)
                except:
                    probas = None

                resultats = [{
                    "Avis": avis_liste[i],
                    "Avis nettoyé": avis_nettoyes[i],
                    "Polarité": f"{emojis[labels[p]]} {labels[p]}",
                    "Confiance": f"{round(max(probas[i])*100)}%" if probas is not None else "N/A"
                } for i, p in enumerate(preds)]

                df_resultats = pd.DataFrame(resultats)
                st.session_state["df_resultats"] = df_resultats

                def extraire_confiance(val):
                    try:
                        return int(val.replace("%", ""))
                    except:
                        return -1

                df_resultats["Confiance_int"] = df_resultats["Confiance"].apply(extraire_confiance)
                df_filtré = df_resultats[df_resultats["Confiance_int"] >= seuil]

            # --- Affichage des résultats en pleine largeur, hors colonnes ---
            if df_filtré.empty:
                st.warning("⚠️ Aucun avis ne dépasse le seuil de confiance choisi.")
            else:
                # --- Affichage du tableau de polarité ---
                st.subheader("Résultats de polarité des avis")
                st.dataframe(df_filtré.drop(columns=["Confiance_int", "Avis nettoyé"]), use_container_width=True)
                
                # Nuages de mots par polarité
                col_pos, col_neg = st.columns(2)
                with col_pos:
                    texte_pos = " ".join(df_filtré[df_filtré["Polarité"].str.contains("Positif")]["Avis nettoyé"])
                    if texte_pos.strip():
                        generer_nuage(texte_pos, "Positif")

                with col_neg:
                    texte_neg = " ".join(df_filtré[df_filtré["Polarité"].str.contains("Négatif")]["Avis nettoyé"])
                    if texte_neg.strip():
                        generer_nuage(texte_neg, "Négatif")
                
               # --- Box Plot des scores de confiance ---
                if "Confiance_int" in df_filtré.columns:
                    st.subheader("Distribution des scores de confiance")
                    fig_box = px.box(df_filtré, y="Confiance_int", color="Polarité", 
                                    color_discrete_map=colors, 
                                    labels={"Confiance_int": "Score de confiance (%)"})
                    fig_box.update_layout(showlegend=False)
                    st.plotly_chart(fig_box, use_container_width=True)


# Onglet Evaluation des modèles
with tab_evaluation:
    if st.session_state.get("avis_csv") and st.session_state.get("labels_csv"):
        avis_net = [nettoyer_texte(a) for a in st.session_state["avis_csv"]]
        y_true = st.session_state["labels_csv"]
        vect_test = vectorizer.transform(avis_net)

        st.subheader("Résumé des performances")
        comparaison = []
        for nom, mod in models.items():
            y_pred = mod.predict(vect_test)
            comparaison.append({
                "Modèle": nom,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Précision": precision_score(y_true, y_pred, zero_division=0),
                "Rappel": recall_score(y_true, y_pred, zero_division=0),
                "F1-score": f1_score(y_true, y_pred, zero_division=0)
            })
        df_comp = pd.DataFrame(comparaison).set_index("Modèle")
        st.dataframe(df_comp.style.format("{:.2%}"), use_container_width=True)

        st.subheader("Matrices de confusion")
        for nom, mod in models.items():
            y_pred = mod.predict(vect_test)
            cm = confusion_matrix(y_true, y_pred)
            df_cm = pd.DataFrame(cm, index=["Négatif", "Positif"], columns=["Prédit Nég", "Prédit Pos"])
            fig = px.imshow(df_cm, text_auto=True, aspect="auto", color_continuous_scale="Blues")
            fig.update_layout(title_text=f"{nom}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Veuillez analyser des avis à partir d’un fichier contenant les colonnes `review` et `label` pour évaluer les modèles.")

# Onglet Interpretation
with tab_interpretation:
    if st.session_state.get("avis_csv") and st.session_state.get("labels_csv"):
        avis_net = [nettoyer_texte(a) for a in st.session_state["avis_csv"]]
        y_true = st.session_state["labels_csv"]
        vect_test = vectorizer.transform(avis_net)

        # Calcul des métriques
        resultats = []
        for nom, mod in models.items():
            y_pred = mod.predict(vect_test)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            resultats.append({
                "Modèle": nom,
                "Accuracy": acc,
                "F1-score": f1
            })

        df_resultats = pd.DataFrame(resultats).set_index("Modèle")
        meilleur = df_resultats["F1-score"].idxmax()
        f1_best = df_resultats.loc[meilleur, "F1-score"]

        # Récupérer matrice de confusion du meilleur modèle
        cm = confusion_matrix(y_true, models[meilleur].predict(vect_test))
        tn, fp, fn, tp = cm.ravel()

        st.subheader("**Interprétation**")
        for nom, row in df_resultats.iterrows():
            niveau = (
                "Excellente" if row["F1-score"] > 0.9 else
                "Bonne" if row["F1-score"] > 0.75 else
                "Acceptable" if row["F1-score"] > 0.6 else
                "Faible"
            )
            st.markdown(f"- **{nom}** : {niveau} performance "
                        f"(Précision : {row['Accuracy']*100:.2f} %, F1-score : {row['F1-score']*100:.2f} %)\n")

        st.markdown(f"Le meilleur modèle est : **{meilleur}** avec un F1-score de **{f1_best*100:.2f} %**")
        st.markdown("### Matrice de confusion du meilleur modèle")
        st.markdown(f"- Faux positifs (FP) : {fp}")
        st.markdown(f"- Faux négatifs (FN) : {fn}")

        err = (fp + fn) / cm.sum()
        st.markdown("### Recommandation")
        if err > 0.2:
            st.error("Taux d'erreur élevé, envisagez d'améliorer les données.")
        elif err > 0.1:
            st.warning("Nettoyage ou réglage d'hyperparamètres recommandé.")
        else:
            st.success("Résultats satisfaisants.")
    else:
        st.info("Veuillez d'abord analyser des avis et fournir les labels pour cette section.")
