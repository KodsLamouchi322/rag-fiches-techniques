"""
=============================================================
  MODULE 1 : INGESTION DES DONNÉES
  ✅ VERSION SANS PGVECTOR
  - Lit les PDFs du dossier courant
  - Découpe le texte en fragments (chunks)
  - Génère les embeddings avec all-MiniLM-L6-v2
  - Stocke dans PostgreSQL (vecteur en TEXT JSON)
=============================================================
"""

import os
import json
import fitz  # PyMuPDF
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PDF_FOLDER = os.path.dirname(os.path.abspath(__file__))
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MODEL_NAME = "all-MiniLM-L6-v2"

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

def lire_pdf(pdf_path: str) -> str:
    """Extrait tout le texte d'un fichier PDF."""
    texte = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            texte += page.get_text()
        doc.close()
    except Exception as e:
        print(f"  ⚠ Erreur lecture {pdf_path}: {e}")
    return texte.strip()


def decouper_en_chunks(texte: str, taille: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe un texte en fragments de taille fixe avec chevauchement."""
    chunks = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille
        chunk = texte[debut:fin].strip()
        if chunk:
            chunks.append(chunk)
        debut += taille - overlap
    return chunks


def connecter_bd():
    """Établit une connexion à PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)


def creer_table(conn):
    """Crée la table embeddings si elle n'existe pas déjà."""
    sql = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id             SERIAL PRIMARY KEY,
        id_document    INT,
        texte_fragment TEXT,
        vecteur        TEXT
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("✅ Table 'embeddings' prête.")


def inserer_fragments(conn, id_document: int, fragments: list[str], vecteurs: np.ndarray):
    """Insère les fragments et leurs vecteurs (en JSON) dans la base."""
    sql = """
        INSERT INTO embeddings (id_document, texte_fragment, vecteur)
        VALUES (%s, %s, %s)
    """
    with conn.cursor() as cur:
        for fragment, vecteur in zip(fragments, vecteurs):
            # Convertir le vecteur numpy en JSON string
            vecteur_json = json.dumps(vecteur.tolist())
            cur.execute(sql, (id_document, fragment, vecteur_json))
    conn.commit()


def main():
    print("=" * 60)
    print("  INGESTION DES FICHES TECHNIQUES ENZYMES")
    print("=" * 60)

    # 1. Charger le modèle
    print(f"\n📦 Chargement du modèle '{MODEL_NAME}'...")
    modele = SentenceTransformer(MODEL_NAME)
    print("✅ Modèle chargé.")

    # 2. Connexion BD
    print("\n🔌 Connexion à PostgreSQL...")
    try:
        conn = connecter_bd()
        print("✅ Connexion établie.")
    except Exception as e:
        print(f"❌ Erreur de connexion : {e}")
        print("\n👉 Vérifiez votre fichier .env")
        return

    # 3. Créer la table
    creer_table(conn)

    # 4. Parcourir les PDFs
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    print(f"\n📂 {len(pdfs)} PDFs trouvés dans le dossier.\n")

    total_chunks = 0

    for id_doc, nom_pdf in enumerate(pdfs, start=1):
        chemin = os.path.join(PDF_FOLDER, nom_pdf)
        print(f"[{id_doc}/{len(pdfs)}] 📄 {nom_pdf}")

        texte = lire_pdf(chemin)
        if not texte:
            print("    ⚠ Aucun texte extrait, fichier ignoré.")
            continue

        chunks = decouper_en_chunks(texte)
        print(f"    ✂  {len(chunks)} fragments")

        vecteurs = modele.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        print(f"    🔢 {len(vecteurs)} vecteurs (dim=384)")

        inserer_fragments(conn, id_doc, chunks, vecteurs)
        print(f"    💾 Inséré en base.")

        total_chunks += len(chunks)

    conn.close()
    print("\n" + "=" * 60)
    print(f"✅ INGESTION TERMINÉE : {total_chunks} fragments au total")
    print("=" * 60)


if __name__ == "__main__":
    main()
