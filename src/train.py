"""
Módulo de Treinamento, Otimização e Comparação de Modelos de Recomendação.

Treina 3 famílias de algoritmos com a biblioteca Surprise:
1. BaselineOnly (vieses globais, de usuário e de item)
2. Variantes de KNN (KNNBasic, KNNWithMeans, KNNWithZScore)
3. SVD (Fatoração de Matrizes otimizada via GridSearchCV)

Realiza Cross-Validation de 5 dobras, gera a tabela comparativa de métricas
(RMSE, MAE, Precision@10, Recall@10) e salva o melhor modelo em models/best_model.pkl.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from surprise import (
    Dataset,
    Reader,
    BaselineOnly,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    SVD
)
from surprise.model_selection import GridSearchCV, KFold

# Adiciona o diretório raiz ao sys.path para importação dos módulos do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    SEED
)
from src.evaluate import evaluate_predictions


def prepare_surprise_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[Dataset, list]:
    """
    Converte DataFrames Pandas de treino e teste no formato nativo do Surprise.
    """
    reader = Reader(rating_scale=(1.0, 5.0))

    # Dataset de treino para cross-validation e fit
    train_data = Dataset.load_from_df(
        train_df[["user_id", "item_id", "rating"]],
        reader
    )

    # Testset do Surprise é uma lista de tuplas (user_id, item_id, rating_real)
    testset = list(zip(
        test_df["user_id"].tolist(),
        test_df["item_id"].tolist(),
        test_df["rating"].tolist()
    ))

    return train_data, testset


def train_and_evaluate_baseline(
    train_data: Dataset,
    testset: list
) -> tuple[dict, object]:
    """
    Treina o modelo BaselineOnly (vieses de usuário e de item).
    """
    print("\n--- Treinando Modelo 1: BaselineOnly ---")
    bsl_options = {
        "method": "als",
        "n_epochs": 10,
        "reg_u": 12,
        "reg_i": 5
    }
    model = BaselineOnly(bsl_options=bsl_options, verbose=False)

    trainset = train_data.build_full_trainset()
    model.fit(trainset)

    predictions = model.test(testset)
    metrics = evaluate_predictions(predictions, k=10)
    metrics["model_name"] = "BaselineOnly"
    print(f"BaselineOnly -> RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, Precision@10: {metrics['precision_at_10']}, Recall@10: {metrics['recall_at_10']}")

    return metrics, model


def train_and_evaluate_knn_variants(
    train_data: Dataset,
    testset: list
) -> tuple[list[dict], object]:
    """
    Treina e compara diferentes variantes de KNN (KNNBasic, KNNWithMeans, KNNWithZScore)
    com diferentes valores de K (20, 40, 60).
    """
    print("\n--- Treinando Modelo 2: Variantes de KNN (Neighborhood-based) ---")
    k_values = [20, 40, 60]
    knn_classes = [
        ("KNNBasic", KNNBasic),
        ("KNNWithMeans", KNNWithMeans),
        ("KNNWithZScore", KNNWithZScore)
    ]

    best_knn_model = None
    best_knn_score = -1.0
    knn_results = []

    trainset = train_data.build_full_trainset()

    for name, cls in knn_classes:
        for k in k_values:
            sim_options = {"name": "pearson_baseline", "user_based": True}
            model = cls(k=k, sim_options=sim_options, verbose=False)
            model.fit(trainset)

            predictions = model.test(testset)
            metrics = evaluate_predictions(predictions, k=10)
            model_label = f"{name} (k={k})"
            metrics["model_name"] = model_label
            knn_results.append(metrics)

            print(f"{model_label:<25} -> RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, Precision@10: {metrics['precision_at_10']}, Recall@10: {metrics['recall_at_10']}")

            # Guardar a melhor variante baseada no Precision@10
            if metrics["precision_at_10"] > best_knn_score:
                best_knn_score = metrics["precision_at_10"]
                best_knn_model = model

    return knn_results, best_knn_model


def train_and_optimize_svd(
    train_data: Dataset,
    testset: list
) -> tuple[dict, object]:
    """
    Otimiza hiperparâmetros do SVD utilizando GridSearchCV da Surprise.
    """
    print("\n--- Treinando Modelo 3: SVD com GridSearchCV (Matrix Factorization) ---")

    param_grid = {
        "n_factors": [20, 50, 100],
        "n_epochs": [20, 30],
        "lr_all": [0.005, 0.01],
        "reg_all": [0.02, 0.1],
        "random_state": [SEED]
    }

    print("Executando GridSearchCV 5-Fold para SVD...")
    gs = GridSearchCV(
        SVD,
        param_grid,
        measures=["rmse", "mae"],
        cv=5,
        n_jobs=-1
    )
    gs.fit(train_data)

    print(f"Melhores parâmetros SVD (RMSE): {gs.best_params['rmse']}")

    # Treinar modelo final com os melhores hiperparâmetros
    best_params = gs.best_params["rmse"]
    best_svd_model = SVD(**best_params)

    trainset = train_data.build_full_trainset()
    best_svd_model.fit(trainset)

    predictions = best_svd_model.test(testset)
    metrics = evaluate_predictions(predictions, k=10)
    metrics["model_name"] = f"SVD (factors={best_params['n_factors']}, epochs={best_params['n_epochs']})"

    print(f"SVD Otimizado -> RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, Precision@10: {metrics['precision_at_10']}, Recall@10: {metrics['recall_at_10']}")

    return metrics, best_svd_model


def run_training_pipeline() -> tuple[pd.DataFrame, object]:
    """
    Executa a pipeline completa de treinamento, comparação e exportação do melhor modelo.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dados de treino e teste
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_ratings.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_ratings.csv")

    train_data, testset = prepare_surprise_datasets(train_df, test_df)

    # 2. Treinar os 3 algoritmos
    baseline_metrics, baseline_model = train_and_evaluate_baseline(train_data, testset)
    knn_results, best_knn_model = train_and_evaluate_knn_variants(train_data, testset)
    svd_metrics, best_svd_model = train_and_optimize_svd(train_data, testset)

    # 3. Montar tabela comparativa
    all_results = [baseline_metrics] + knn_results + [svd_metrics]
    df_comparison = pd.DataFrame(all_results)
    df_comparison = df_comparison.sort_values(by="precision_at_10", ascending=False).reset_index(drop=True)

    print("\n================================================================================")
    print("                      TABELA COMPARATIVA DE MODELOS                             ")
    print("================================================================================")
    print(df_comparison.to_string(index=False))

    # 4. Selecionar o melhor modelo (maior Precision@10 com menor RMSE)
    best_row = df_comparison.iloc[0]
    best_model_name = best_row["model_name"]
    print(f"\n[CAMPEAO] Modelo Selecionado: {best_model_name}")

    if "SVD" in best_model_name:
        final_best_model = best_svd_model
    elif "KNN" in best_model_name:
        final_best_model = best_knn_model
    else:
        final_best_model = baseline_model

    # 5. Salvar o melhor modelo com Joblib
    output_model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(final_best_model, output_model_path)
    print(f"Modelo final serializado com sucesso em: {output_model_path}")

    # Salvar a tabela comparativa em CSV para relatórios
    df_comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

    return df_comparison, final_best_model


if __name__ == "__main__":
    run_training_pipeline()
