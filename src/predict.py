"""
Módulo de Predição e Geração de Recomendações Integrado à Explicabilidade.

Fornece a API de inferência do modelo para:
1. Prever a nota de um usuário para um filme específico.
2. Gerar a lista de Top-N recomendações não vistas, acopladas com explicações em linguagem natural.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

# Adiciona o diretório raiz ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.explainer import explain

_MODEL = None
_RATINGS_DF = None
_MOVIES_DF = None


def _load_prediction_resources():
    """Carrega o modelo e os DataFrames em memória para inferências rápidas."""
    global _MODEL, _RATINGS_DF, _MOVIES_DF
    if _MODEL is None:
        _MODEL = joblib.load(MODELS_DIR / "best_model.pkl")
    if _RATINGS_DF is None:
        _RATINGS_DF = pd.read_csv(PROCESSED_DATA_DIR / "processed_ratings.csv")
    if _MOVIES_DF is None:
        _MOVIES_DF = pd.read_csv(PROCESSED_DATA_DIR / "processed_movies.csv")


def predict_rating(user_id: int, movie_id: int, with_explanation: bool = False) -> dict:
    """
    Prevê a nota estimada para um par (user_id, movie_id) e opcionalmente inclui a explicação.
    """
    _load_prediction_resources()
    pred = _MODEL.predict(user_id, movie_id)
    predicted_rating = round(float(pred.est), 2)

    result = {
        "user_id": user_id,
        "movie_id": movie_id,
        "predicted_rating": predicted_rating
    }

    if with_explanation:
        explanation_data = explain(user_id, movie_id)
        result["explanation"] = explanation_data

    return result


def get_top_n_recommendations(
    user_id: int,
    n: int = 10,
    include_explanations: bool = True
) -> list[dict]:
    """
    Gera o ranking de Top-N filmes não assistidos pelo usuário, com explicações.

    Passos:
    1. Identificar o conjunto de filmes que o usuário ainda não avaliou.
    2. Calcular a nota estimada para cada filme candidato.
    3. Ordenar e selecionar os N filmes com maior nota prevista.
    4. Integrar o módulo explainer.py para cada item do Top-N.
    """
    _load_prediction_resources()

    # Identificar filmes já vistos pelo usuário
    user_seen_movies = set(_RATINGS_DF[_RATINGS_DF["user_id"] == user_id]["item_id"])

    # Filmes candidatos (todos os filmes processados menos os vistos)
    all_movies = set(_MOVIES_DF["movie_id"])
    candidate_movies = list(all_movies - user_seen_movies)

    if not candidate_movies:
        return []

    # Calcular estimativas para todos os candidatos
    predictions = []
    for m_id in candidate_movies:
        est = _MODEL.predict(user_id, m_id).est
        predictions.append((m_id, est))

    # Ordenar por nota estimada em ordem decrescente
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_candidates = predictions[:n]

    recommendations = []
    for m_id, est in top_n_candidates:
        movie_row = _MOVIES_DF[_MOVIES_DF["movie_id"] == m_id]
        title = movie_row["movie_title"].values[0] if not movie_row.empty else f"Filme #{m_id}"

        rec_item = {
            "movie_id": m_id,
            "title": title,
            "predicted_rating": round(float(est), 2)
        }

        if include_explanations:
            rec_item["explanation"] = explain(user_id, m_id)

        recommendations.append(rec_item)

    return recommendations


if __name__ == "__main__":
    print("=== TESTANDO PIPELINE DE PREDIÇÃO E RECOMENDAÇÃO ===")
    recs = get_top_n_recommendations(user_id=1, n=3, include_explanations=True)
    for idx, r in enumerate(recs, 1):
        print(f"\n--- Recomendação #{idx}: {r['title']} (Nota Estimada: {r['predicted_rating']}) ---")
        print(f"Explicação: {r['explanation']['explanation_text']}")
