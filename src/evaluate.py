"""
Módulo de Avaliação de Desempenho e Métricas de Recomendação.

Implementa o cálculo de:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Precision@K (Precisão no Top-K)
- Recall@K (Revocação no Top-K)

Seguindo os princípios SOLID (Single Responsibility), este módulo foca
exclusivamente na computação estatística das métricas de desempenho.
"""

from collections import defaultdict
import numpy as np
from surprise import accuracy


def calculate_rmse(predictions: list) -> float:
    """Calcula a raiz do erro quadrático médio (RMSE)."""
    return accuracy.rmse(predictions, verbose=False)


def calculate_mae(predictions: list) -> float:
    """Calcula o erro absoluto médio (MAE)."""
    return accuracy.mae(predictions, verbose=False)


def precision_recall_at_k(
    predictions: list,
    k: int = 10,
    threshold: float = 4.0
) -> tuple[float, float]:
    """
    Calcula o Precision@K e Recall@K médios sobre todos os usuários.

    Definição:
    - Item Relevante: Item cuja nota real (r_ui) é >= `threshold` (default: 4.0).
    - Item Recomendado: Itens com as K maiores notas previstas (est) para o usuário.

    Fórmulas:
    - Precision@K = (Itens Recomendados no Top-K que são Relevantes) / K
    - Recall@K = (Itens Recomendados no Top-K que são Relevantes) / (Total de Itens Relevantes do Usuário)
    """
    # Mapear predições por usuário: user_id -> [(est, r_ui), ...]
    user_est_true = defaultdict(list)
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        # Ordenar itens do usuário pela estimativa (nota prevista) em ordem decrescente
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Total de itens relevantes do usuário (nota real >= threshold)
        n_rel = sum((r_ui >= threshold) for (_, r_ui) in user_ratings)

        # Top-K itens recomendados
        top_k = user_ratings[:k]

        # Número de itens relevantes dentro do Top-K
        n_rec_k = sum((r_ui >= threshold) for (_, r_ui) in top_k)

        # Precision@K para este usuário
        precisions[uid] = n_rec_k / k

        # Recall@K para este usuário (apenas se ele possuir itens relevantes)
        if n_rel > 0:
            recalls[uid] = n_rec_k / n_rel

    # Média global entre todos os usuários
    mean_precision = float(np.mean(list(precisions.values()))) if precisions else 0.0
    mean_recall = float(np.mean(list(recalls.values()))) if recalls else 0.0

    return mean_precision, mean_recall


def evaluate_predictions(
    predictions: list,
    k: int = 10,
    threshold: float = 4.0
) -> dict[str, float]:
    """
    Avalia um conjunto de predições do Surprise e retorna todas as 4 métricas.

    Returns:
        dict: {'rmse': ..., 'mae': ..., 'precision_at_k': ..., 'recall_at_k': ...}
    """
    rmse_val = calculate_rmse(predictions)
    mae_val = calculate_mae(predictions)
    prec_k, rec_k = precision_recall_at_k(predictions, k=k, threshold=threshold)

    return {
        "rmse": round(rmse_val, 4),
        "mae": round(mae_val, 4),
        f"precision_at_{k}": round(prec_k, 4),
        f"recall_at_{k}": round(rec_k, 4)
    }
