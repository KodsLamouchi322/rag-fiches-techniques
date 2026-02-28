-- ============================================================
--  SCRIPT DE PRÉPARATION DE LA BASE DE DONNÉES
--  ✅ VERSION SANS PGVECTOR — PostgreSQL standard seulement
-- ============================================================

-- 1. Créer la base de données
--    Dans pgAdmin : clic droit sur "Databases" → Create → Database
--    Nom : enzymes_db

-- 2. Se connecter à enzymes_db puis exécuter ce script :

-- 3. Créer la table (vecteur stocké en TEXT, pas besoin de pgvector !)
CREATE TABLE IF NOT EXISTS embeddings (
    id             SERIAL PRIMARY KEY,
    id_document    INT,
    texte_fragment TEXT,
    vecteur        TEXT    -- stocké comme JSON : "[0.1, 0.2, ...]"
);

-- Vérification
SELECT COUNT(*) AS total_fragments FROM embeddings;
