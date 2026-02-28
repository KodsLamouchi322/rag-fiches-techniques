"""
=============================================================
  MODULE 2 : RECHERCHE SÉMANTIQUE (RAG)
  ✅ VERSION SANS PGVECTOR
  
  Fonctionnement :
  1. Reçoit une question utilisateur
  2. Génère l'embedding de la question (all-MiniLM-L6-v2)
  3. Récupère tous les vecteurs depuis PostgreSQL
  4. Calcule la similarité cosinus en Python (sklearn)
  5. Retourne les Top K=3 fragments les plus pertinents
=============================================================
"""

import os
import json
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "enzymes_db"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


# ─────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────

def connecter_bd():
    return psycopg2.connect(**DB_CONFIG)


def recuperer_tous_les_embeddings(conn):
    """
    Récupère tous les fragments et leurs vecteurs depuis la base.
    Les vecteurs sont stockés en JSON → on les reparse en numpy.
    """
    sql = "SELECT id, texte_fragment, vecteur FROM embeddings;"

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    if not rows:
        return [], [], np.array([])

    ids, fragments, vecteurs = [], [], []

    for row in rows:
        id_, texte, vecteur_json = row
        ids.append(id_)
        fragments.append(texte)
        # Convertir le JSON string "[0.1, 0.2, ...]" en liste Python
        vecteur_liste = json.loads(vecteur_json)
        vecteurs.append(vecteur_liste)

    matrice_vect = np.array(vecteurs, dtype=np.float32)
    return ids, fragments, matrice_vect


def recherche_semantique(question: str, modele: SentenceTransformer, conn) -> list[dict]:
    """
    Cœur du module RAG :
    1. Encode la question
    2. Compare avec tous les fragments (cosine similarity)
    3. Retourne le Top K
    """
    # ÉTAPE 1 : Générer l'embedding de la question
    embedding_question = modele.encode(
        [question],
        normalize_embeddings=True
    )  # shape: (1, 384)

    # ÉTAPE 2 : Récupérer tous les fragments
    ids, fragments, matrice_vect = recuperer_tous_les_embeddings(conn)

    if len(fragments) == 0:
        print("⚠ La base est vide ! Lancez d'abord 01_ingestion.py")
        return []

    # ÉTAPE 3 : Similarité cosinus
    scores = cosine_similarity(embedding_question, matrice_vect)[0]

    # ÉTAPE 4 : Trier par score décroissant → Top K
    indices_tries = np.argsort(scores)[::-1][:TOP_K]

    # ÉTAPE 5 : Construire les résultats
    resultats = []
    for rang, idx in enumerate(indices_tries, start=1):
        resultats.append({
            "rang":  rang,
            "texte": fragments[idx],
            "score": float(scores[idx]),
            "id":    ids[idx],
        })

    return resultats


def afficher_resultats(resultats: list[dict], question: str):
    """Affiche les résultats de manière claire."""
    print("\n" + "═" * 70)
    print(f"  🔍 QUESTION : {question}")
    print("═" * 70)

    if not resultats:
        print("  Aucun résultat trouvé.")
        return

    for res in resultats:
        print(f"\n  📌 Résultat {res['rang']}")
        print(f"  {'─' * 66}")
        print(f"  📄 Texte :")
        texte = res['texte']
        for i in range(0, len(texte), 80):
            print(f"     {texte[i:i+80]}")
        print(f"\n  🎯 Score de similarité : {res['score']:.4f}")

    print("\n" + "═" * 70)


def main():
    print("=" * 70)
    print("  MODULE DE RECHERCHE SÉMANTIQUE - RAG (Boulangerie & Pâtisserie)")
    print("=" * 70)

    # 1. Charger le modèle
    print(f"\n📦 Chargement du modèle '{MODEL_NAME}'...")
    modele = SentenceTransformer(MODEL_NAME)
    print("✅ Modèle prêt.")

    # 2. Connexion BD
    print("\n🔌 Connexion à PostgreSQL...")
    try:
        conn = connecter_bd()
        print("✅ Connexion établie.")
    except Exception as e:
        print(f"❌ Erreur de connexion : {e}")
        print("👉 Vérifiez votre fichier .env")
        return

    # 3. Boucle interactive
    print("\n💡 Entrez vos questions (tapez 'quitter' pour arrêter)\n")

    while True:
        question = input("❓ Votre question : ").strip()

        if question.lower() in ["quitter", "quit", "exit", "q"]:
            print("\n👋 Au revoir !")
            break

        if not question:
            print("  ⚠ Question vide.\n")
            continue

        resultats = recherche_semantique(question, modele, conn)
        afficher_resultats(resultats, question)
        print()

    conn.close()


if __name__ == "__main__":
    main()
