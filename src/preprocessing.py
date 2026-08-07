"""
Módulo de Pré-processamento e Preparação de Dados.

Este módulo é responsável por:
1. Tratar valores ausentes e inconsistências nos DataFrames brutos.
2. Filtrar iterativamente usuários e filmes abaixo dos limiares mínimos (k-core filtering).
3. Realizar a divisão em conjuntos de Treino (80%) e Teste (20%) estratificada por usuário.
4. Serializar os artefatos processados no diretório data/processed.

Todas as decisões técnicas de limpeza contêm comentários justificando a sua escolha (Princípios Clean Code e KISS).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Adiciona o diretório raiz ao sys.path para importação do módulo src.config
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    PROCESSED_DATA_DIR,
    SEED,
    RATINGS_COLUMNS,
    GENRES
)
from src.data_loader import load_movielens_data


def handle_missing_values(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Identifica e trata valores ausentes nos DataFrames de avaliações, filmes e usuários.

    Decisão de Limpeza:
    - Avaliações: Linhas com nulos em 'user_id', 'item_id' ou 'rating' são removidas, pois a matriz
      de utilidade exige registros válidos.
    - Filmes: 'release_date' nulo é preenchido com valor genérico ('Unknown'), pois a falta da data
      não impede o uso dos gêneros para explicabilidade.
    """
    # 1. Tratamento em Avaliações
    initial_ratings_count = len(ratings)
    ratings_clean = ratings.dropna(subset=["user_id", "item_id", "rating"]).copy()
    dropped_ratings = initial_ratings_count - len(ratings_clean)
    if dropped_ratings > 0:
        print(f"[Limpeza] Removidas {dropped_ratings} avaliações com valores ausentes.")

    # Garantir tipos de dados inteiros corretos em ratings
    ratings_clean["user_id"] = ratings_clean["user_id"].astype(int)
    ratings_clean["item_id"] = ratings_clean["item_id"].astype(int)
    ratings_clean["rating"] = ratings_clean["rating"].astype(float)

    # 2. Tratamento em Filmes
    movies_clean = movies.copy()
    if movies_clean["release_date"].isnull().sum() > 0:
        movies_clean["release_date"] = movies_clean["release_date"].fillna("01-Jan-1900")
        print("[Limpeza] Preenchidos valores ausentes em 'release_date' dos filmes.")

    # Fillna em colunas de texto não essenciais
    movies_clean["video_release_date"] = movies_clean["video_release_date"].fillna("")
    movies_clean["IMDb_URL"] = movies_clean["IMDb_URL"].fillna("")

    # 3. Tratamento em Usuários
    users_clean = users.copy()

    return ratings_clean, movies_clean, users_clean


def filter_k_core(
    df_ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 10
) -> pd.DataFrame:
    """
    Aplica a filtragem iterativa k-core para garantir que:
    - Cada usuário tenha no mínimo `min_user_ratings` avaliações (default: 20).
    - Cada filme tenha no mínimo `min_movie_ratings` avaliações (default: 10).

    Decisão Técnica:
    Utilizamos um loop 'while' porque a remoção de um filme com poucas avaliações pode
    fazer com que o total de avaliações de determinado usuário caia abaixo do limiar
    mínimo, exigindo filtragem recíproca iterativa até a estabilização.
    """
    df_filtered = df_ratings.copy()
    iteration = 0

    while True:
        iteration += 1
        initial_len = len(df_filtered)

        # Contar avaliações por usuário
        user_counts = df_filtered["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df_filtered = df_filtered[df_filtered["user_id"].isin(valid_users)]

        # Contar avaliações por filme
        movie_counts = df_filtered["item_id"].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df_filtered = df_filtered[df_filtered["item_id"].isin(valid_movies)]

        # Critério de parada: quando nenhuma linha adicional for removida na iteração
        if len(df_filtered) == initial_len:
            break

    print(
        f"[Filtragem k-core] Concluída em {iteration} iterações. "
        f"Registros finais: {len(df_filtered):,} (usuários >= {min_user_ratings}, filmes >= {min_movie_ratings})."
    )
    return df_filtered


def split_train_test_stratified_by_user(
    df_ratings: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o conjunto de avaliações em Treino (80%) e Teste (20%), garantindo que a
    estratificação ocorra POR USUÁRIO (per-user stratified split).

    Decisão Arquitetural:
    Uma divisão aleatória simples (random split) poderia deixar usuários sem avaliações
    no conjunto de treino ou sem avaliações de teste. Ao agrupar por 'user_id' e aplicar
    80/20 em cada usuário individualmente, garantimos que TODO usuário possua histórico de
    treinamento e alvos no conjunto de teste para avaliação personalizada.
    """
    train_list = []
    test_list = []

    # Agrupar por usuário e dividir 80/20 individualmente
    for user_id, group in df_ratings.groupby("user_id"):
        if len(group) < 2:
            # Caso extremo de segurança se o usuário tiver apenas 1 avaliação
            train_list.append(group)
            continue

        train_u, test_u = train_test_split(
            group,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        train_list.append(train_u)
        test_list.append(test_u)

    df_train = pd.concat(train_list, ignore_index=True)
    df_test = pd.concat(test_list, ignore_index=True)

    # Reordenar de forma aleatória preservando a reproduzibilidade
    df_train = df_train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df_test = df_test.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    actual_test_ratio = len(df_test) / len(df_ratings)
    print(
        f"[Split Treino/Teste] Treino: {len(df_train):,} linhas ({100*(1-actual_test_ratio):.1f}%) | "
        f"Teste: {len(df_test):,} linhas ({100*actual_test_ratio:.1f}%)"
    )

    return df_train, df_test


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR
) -> None:
    """
    Serializa os DataFrames processados em arquivos CSV no diretório data/processed/.

    Decisão de Persistência:
    Utiliza CSV com index=False para garantir portabilidade e facilidade de inspeção.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train_ratings.csv", index=False)
    test_df.to_csv(output_dir / "test_ratings.csv", index=False)
    full_df.to_csv(output_dir / "processed_ratings.csv", index=False)
    movies_df.to_csv(output_dir / "processed_movies.csv", index=False)
    users_df.to_csv(output_dir / "processed_users.csv", index=False)

    print(f"[Persistência] Todos os arquivos salvos com sucesso em: {output_dir}")


def run_preprocessing_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executa a pipeline completa de pré-processamento de dados.

    Returns:
        tuple: (train_ratings, test_ratings, movies_clean, users_clean)
    """
    print("=== INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO ===")

    # 1. Carregar dados brutos
    raw_ratings, raw_movies, raw_users = load_movielens_data()
    print(f"Dados brutos carregados: {len(raw_ratings):,} avaliações.")

    # 2. Tratar nulos
    ratings_clean, movies_clean, users_clean = handle_missing_values(raw_ratings, raw_movies, raw_users)

    # 3. Filtragem k-core (usuários >= 20, filmes >= 10)
    ratings_filtered = filter_k_core(ratings_clean, min_user_ratings=20, min_movie_ratings=10)

    # Filtrar também os DataFrames de filmes e usuários para manter consistência relacional
    valid_movie_ids = set(ratings_filtered["item_id"])
    valid_user_ids = set(ratings_filtered["user_id"])

    movies_clean = movies_clean[movies_clean["movie_id"].isin(valid_movie_ids)].reset_index(drop=True)
    users_clean = users_clean[users_clean["user_id"].isin(valid_user_ids)].reset_index(drop=True)

    # 4. Divisão Treino / Teste estratificada por usuário
    train_ratings, test_ratings = split_train_test_stratified_by_user(
        ratings_filtered,
        test_size=0.20,
        random_state=SEED
    )

    # 5. Salvar em data/processed/
    save_processed_data(
        train_df=train_ratings,
        test_df=test_ratings,
        full_df=ratings_filtered,
        movies_df=movies_clean,
        users_df=users_clean,
        output_dir=PROCESSED_DATA_DIR
    )

    print("=== PIPELINE DE PRÉ-PROCESSAMENTO CONCLUÍDA COM SUCESSO! ===")
    return train_ratings, test_ratings, movies_clean, users_clean


if __name__ == "__main__":
    run_preprocessing_pipeline()
