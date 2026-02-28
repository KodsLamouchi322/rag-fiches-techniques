"""
=============================================================
  PROTOTYPE RAG — Recherche Sémantique Avancée
  Module de recherche sémantique pour fiches techniques
  (boulangerie & patisserie)
=============================================================
"""

import os
import json
import time
import re
import csv
import io
import psycopg2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "enzymes_db"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

PDF_FOLDER = os.path.dirname(os.path.abspath(__file__))

print(f"Chargement du modele '{MODEL_NAME}'...")
modele = SentenceTransformer(MODEL_NAME)
print("Modele pret.")

_cache = {"ids": None, "fragments": None, "vecteurs": None, "doc_ids": None}
historique = []
doc_names = {}


def connecter_bd():
    return psycopg2.connect(**DB_CONFIG)


def charger_doc_names():
    global doc_names
    if doc_names:
        return doc_names
    pdfs = sorted([f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")])
    for i, nom in enumerate(pdfs, start=1):
        doc_names[i] = nom.replace(".pdf", "")
    return doc_names


def charger_embeddings():
    if _cache["ids"] is not None:
        return _cache["ids"], _cache["fragments"], _cache["vecteurs"], _cache["doc_ids"]

    conn = connecter_bd()
    sql = "SELECT id, id_document, texte_fragment, vecteur FROM embeddings;"
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    conn.close()

    if not rows:
        return [], [], np.array([]), []

    ids, doc_ids, fragments, vecteurs = [], [], [], []
    for row in rows:
        id_, doc_id, texte, vecteur_json = row
        ids.append(id_)
        doc_ids.append(doc_id)
        fragments.append(texte)
        vecteurs.append(json.loads(vecteur_json))

    _cache["ids"] = ids
    _cache["doc_ids"] = doc_ids
    _cache["fragments"] = fragments
    _cache["vecteurs"] = np.array(vecteurs, dtype=np.float32)

    return ids, fragments, _cache["vecteurs"], doc_ids


def extraire_mots_cles(question, texte):
    mots_question = set(re.findall(r'\b\w{4,}\b', question.lower()))
    mots_texte = texte.lower()
    return [m for m in mots_question if m in mots_texte]


def nettoyer_texte(texte):
    texte = re.sub(r'\n{3,}', '\n\n', texte)
    texte = re.sub(r'[ \t]+', ' ', texte)
    return texte.strip()


def generer_reformulations(question, resultats):
    """
    Genere des suggestions de reformulation basees sur
    les mots-cles trouves dans les fragments les plus pertinents.
    """
    # Extraire les mots importants de la question
    mots_question = set(re.findall(r'\b\w{4,}\b', question.lower()))
    
    # Extraire les termes techniques des resultats
    termes_resultats = set()
    for r in resultats:
        # Trouver les mots importants dans les fragments
        mots = re.findall(r'\b[A-Za-zÀ-ÿ\-]{4,}\b', r["texte"])
        for m in mots:
            ml = m.lower()
            # Garder les mots techniques qui ne sont pas dans la question
            if ml not in mots_question and ml not in {
                "dans", "pour", "avec", "plus", "cette", "sont",
                "entre", "selon", "comme", "aussi", "autre", "leurs",
                "tout", "tous", "elle", "elles", "nous", "vous",
                "leur", "notre", "votre", "avoir", "etre", "faire",
                "very", "that", "this", "with", "from", "they", "have",
                "been", "were", "will", "would", "could", "should"
            }:
                termes_resultats.add(ml)
    
    # Termes techniques specifiques au domaine
    termes_domaine = []
    domaine_keywords = [
        "amylase", "xylanase", "lipase", "protease", "glucose",
        "oxydase", "transglutaminase", "maltogenic", "ascorbique",
        "dosage", "enzyme", "farine", "panification", "fermentation",
        "temperature", "humidite", "conservation", "shelf",
        "freshness", "softness", "extensibilite", "pate",
        "boulangerie", "patisserie", "ameliorant"
    ]
    for t in termes_resultats:
        for kw in domaine_keywords:
            if kw in t or t in kw:
                termes_domaine.append(t)
                break
    
    # Generer 3 reformulations
    suggestions = []
    
    # Suggestion 1: Plus specifique avec termes techniques
    if termes_domaine:
        termes = list(set(termes_domaine))[:3]
        suggestions.append(f"Quelle est la fonction et le dosage de {', '.join(termes)} ?")
    
    # Suggestion 2: Question plus precise
    if mots_question:
        mots_principaux = [m for m in mots_question if len(m) > 4][:2]
        if mots_principaux:
            suggestions.append(
                f"Quelles sont les recommandations d'utilisation pour {' et '.join(mots_principaux)} en boulangerie ?"
            )
    
    # Suggestion 3: Question orientee application
    suggestions.append(
        "Quels sont les dosages recommandes des enzymes pour la panification ?"
    )
    
    # Suggestion 4: Question orientee produit
    if termes_domaine:
        suggestions.append(
            f"Fiche technique et proprietes de {termes_domaine[0]}"
        )
    
    # Deduplicate et limiter a 3
    seen = set()
    unique = []
    for s in suggestions:
        sl = s.lower()
        if sl not in seen and sl != question.lower():
            seen.add(sl)
            unique.append(s)
    
    return unique[:3]


def analyser_qualite(scores_top, question, resultats):
    if not scores_top:
        return {"niveau": "inconnu", "message": "Aucun resultat", "reformulations": []}

    top = scores_top[0]
    reformulations = []
    
    if top >= 0.65:
        return {
            "niveau": "excellent",
            "message": "Correspondance forte. Les fragments retrouves sont tres pertinents.",
            "reformulations": []
        }
    elif top >= 0.50:
        return {
            "niveau": "bon",
            "message": "Bonne correspondance. Les fragments couvrent probablement le sujet.",
            "reformulations": []
        }
    elif top >= 0.35:
        reformulations = generer_reformulations(question, resultats)
        return {
            "niveau": "moyen",
            "message": "Correspondance partielle. Essayez une des reformulations suggerees ci-dessous.",
            "reformulations": reformulations
        }
    else:
        reformulations = generer_reformulations(question, resultats)
        return {
            "niveau": "faible",
            "message": "Faible correspondance. La base ne contient peut-etre pas d'information directe. Essayez ces reformulations :",
            "reformulations": reformulations
        }


def recherche_semantique(question: str, top_k: int = TOP_K) -> dict:
    debut = time.time()

    embedding_question = modele.encode([question], normalize_embeddings=True)
    ids, fragments, matrice_vect, doc_ids = charger_embeddings()

    if len(fragments) == 0:
        return {"resultats": [], "temps_ms": 0, "total_fragments": 0}

    scores = cosine_similarity(embedding_question, matrice_vect)[0]
    indices_tries = np.argsort(scores)[::-1]

    doc_name_map = charger_doc_names()
    resultats = []
    for rang, idx in enumerate(indices_tries[:top_k], start=1):
        texte_nettoye = nettoyer_texte(fragments[idx])
        mots_cles = extraire_mots_cles(question, texte_nettoye)
        doc_id = doc_ids[idx]
        resultats.append({
            "rang":        rang,
            "texte":       texte_nettoye,
            "score":       round(float(scores[idx]), 4),
            "id":          ids[idx],
            "mots_cles":   mots_cles,
            "document":    doc_name_map.get(doc_id, f"Document {doc_id}"),
            "id_document": doc_id,
        })

    temps_ms = round((time.time() - debut) * 1000, 1)

    qualite = analyser_qualite(
        [r["score"] for r in resultats],
        question,
        resultats
    )

    return {
        "resultats":       resultats,
        "temps_ms":        temps_ms,
        "total_fragments": len(fragments),
        "score_moyen":     round(float(np.mean(scores)), 4),
        "qualite":         qualite,
    }


def get_stats():
    conn = connecter_bd()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM embeddings;")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT id_document) FROM embeddings;")
        docs = cur.fetchone()[0]
        cur.execute("SELECT AVG(LENGTH(texte_fragment)) FROM embeddings;")
        avg_len = cur.fetchone()[0]
    conn.close()
    return {
        "total_fragments": total,
        "total_documents": docs,
        "longueur_moyenne": round(avg_len) if avg_len else 0,
    }


# ── ROUTES ──

@app.route("/")
def index():
    stats = get_stats()
    return render_template("index.html", stats=stats)


@app.route("/recherche", methods=["POST"])
def recherche():
    data = request.get_json()
    question = data.get("question", "").strip()
    top_k = data.get("top_k", TOP_K)

    if not question:
        return jsonify({"erreur": "Question vide"}), 400

    try:
        resultat = recherche_semantique(question, top_k=top_k)
        historique.insert(0, {
            "question":  question,
            "score_top": resultat["resultats"][0]["score"] if resultat["resultats"] else 0,
            "temps_ms":  resultat["temps_ms"],
        })
        if len(historique) > 30:
            historique.pop()
        return jsonify(resultat)
    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


@app.route("/historique")
def get_historique():
    return jsonify(historique)


@app.route("/export/csv", methods=["POST"])
def export_csv():
    data = request.get_json()
    resultats = data.get("resultats", [])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Rang", "Score", "Document", "Texte", "Mots-cles"])
    for r in resultats:
        writer.writerow([r["rang"], r["score"], r.get("document",""), r["texte"], ", ".join(r.get("mots_cles",[]))])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=resultats_rag.csv"}
    )


@app.route("/export/json", methods=["POST"])
def export_json():
    data = request.get_json()
    return Response(
        json.dumps(data, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=resultats_rag.json"}
    )


if __name__ == "__main__":
    print("\nPrototype RAG demarre sur http://localhost:5000\n")
    app.run(debug=True, port=5000)
