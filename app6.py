import streamlit as st
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

############################################
# 1. CONSTANTES
############################################

MOTS_CLES_FIXES = [
    "robotique",
    "IA",
    "automatisation",
    "technologie",
    "innovation",
    "futur",
    "espace",
    "astronautes",
    "trou de ver",
    "exploration",
    "galaxie",
    "mythologie",
    "légendes",
    "symbolisme",
    "esthétique",
    "art",
    "peinture",
    "sculpture",
    "industrie",
    "histoire",
    "psychologie",
    "enquête",
    "scène de crime",
    "manipulation",
    "tension",
    "mystère",
    "récit",
    "philosophie",
    "éthique",
    "pouvoir",
    "identité",
    "énergie",
    "Force",
    "Jedi",
    "Sith",
    "économie",
    "politique",
    "société",
    "culture",
    "créatures",
    "mythes",
    "univers",
    "cosmos",
    "temps",
    "espace-temps",
    "réalité",
    "rêves",
    "mémoire",
    "perception"
]


CATEGORIES_FIXES = [
    "science-fiction",
    "cinéma",
    "policier"
]

############################################
# 2. FONCTIONS DE RECHERCHE / FILTRES
############################################

def recherche_semantique(df, requete, top_k=3):
    corpus = (
        df["titre"].fillna("") + " " +
        df["mots_cles"].fillna("") + " " +
        df["description"].fillna("")
    ).tolist()

    vectorizer = TfidfVectorizer(stop_words="english") #voir pourquoi french ne fonctionne pas ici : english sert à rien sur mes tests français
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vec = vectorizer.transform([requete])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = scores.argsort()[::-1][:top_k]
    resultats = df.iloc[top_indices].copy()
    resultats["score"] = scores[top_indices]

    return resultats


def rechercher_par_categorie(df, categorie):
    return df[df["categorie_finale"].str.lower() == categorie.lower()]


def filtrer_par_mot_cle(df, mot_cle):
    return df[df["mots_cles"].str.contains(mot_cle, case=False, na=False)]


def filtrer_par_mots_cles(df, mots):
    pattern = "|".join(mots)

    return df[
        df["titre"].str.contains(pattern, case=False, na=False)
        | df["mots_cles"].str.contains(pattern, case=False, na=False)
        | df["description"].str.contains(pattern, case=False, na=False)
        | df["categorie_finale"].str.contains(pattern, case=False, na=False)
    ]


def filtrage_combine(df, categorie=None, tags=None):
    result = df.copy()

    # Filtre catégorie
    if categorie:
        result = result[result["categorie_finale"].str.lower() == categorie.lower()]

    # Filtre tags (mots-clés)
    if tags:
        pattern = "|".join(tags)
        result = filtrer_par_mots_cles(df,tags)

    return result


############################################
# 3. APPLICATION STREAMLIT
############################################

st.title("🔎 Outil de recherche documentaire")

# Chargement de la base existante fixe pour les tests de recherches (documents_analyzed_B20 est dynamique car pour ajout)
CSV_PATH = "base20.csv"

if not os.path.exists(CSV_PATH):
    st.error(f"❌ La base de données '{CSV_PATH}' est introuvable.")
    st.stop()

df = pd.read_csv(CSV_PATH)

st.subheader("Base de données chargée")
st.dataframe(df)

st.header("Recherche et filtrage")

menu = st.radio(
    "Choix du mode de recherche",
    [
        "Recherche sémantique",
        "Recherche combinée (catégorie + tags)"
    ]
)

############################################
# 4. MODES DE RECHERCHE
############################################

if menu == "Catégorie":
    categorie = st.selectbox(
        "Catégorie recherchée :",
        [""] + CATEGORIES_FIXES
    )
    if categorie:
        st.dataframe(rechercher_par_categorie(df, categorie))

elif menu == "1 mot-clé":
    mot = st.text_input("Mot-clé")
    if mot:
        st.dataframe(filtrer_par_mot_cle(df, mot))

elif menu == "Plusieurs mots-clés":
    mots = st.text_input("Liste de mots séparés par des virgules")
    if mots:
        liste = [m.strip() for m in mots.split(",") if m.strip()]
        if liste:
            st.dataframe(filtrer_par_mots_cles(df, liste))

elif menu == "Recherche sémantique":
    requete = st.text_input("Recherche sémantique :")
    top_k = st.slider("Nombre de résultats", min_value=1, max_value=10, value=3)
    if requete:
        st.dataframe(recherche_semantique(df, requete, top_k=top_k))

elif menu == "Mots-clés ":
    st.subheader("Sélectionnez des mots-clés")

    selection = st.multiselect(
        "Choisissez un ou plusieurs mots-clés :",
        options=MOTS_CLES_FIXES
    )

    if selection:
        st.write(f"Résultats pour : {', '.join(selection)}")
        st.dataframe(filtrer_par_mots_cles(df, selection))

elif menu == "Recherche combinée (catégorie + tags)":
    st.header("Recherche combinée")

    # Catégorie
    categorie = st.selectbox(
        "Catégorie :",
        [""] + CATEGORIES_FIXES
    )
    categorie = categorie if categorie != "" else None

    # Tags
    tags = st.multiselect(
        "Mots-clés:",
        options=MOTS_CLES_FIXES
    )

    if st.button("Lancer la recherche"):
        resultats = filtrage_combine(df, categorie, tags)
        st.dataframe(resultats)
