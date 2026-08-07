"""
Script para executar o conteúdo das células do notebook 01_exploracao.ipynb
e salvar as saídas de texto e gráficos interativos no arquivo .ipynb.
"""

import json
import sys
from pathlib import Path
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import SEED, GENRES
from src.data_loader import load_movielens_data

NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "01_exploracao.ipynb"


def run_cell_1():
    out = io.StringIO()
    sys.stdout = out
    print("Ambiente configurado com sucesso! Semente global:", SEED)
    sys.stdout = sys.__stdout__
    return out.getvalue()


def run_cell_2():
    out = io.StringIO()
    sys.stdout = out
    ratings, movies, users = load_movielens_data()
    print("=== DIMENSÕES DOS DATASETS ===")
    print(f"Avaliações (ratings): {ratings.shape[0]:,} linhas, {ratings.shape[1]} colunas")
    print(f"Filmes (movies)     : {movies.shape[0]:,} linhas, {movies.shape[1]} colunas")
    print(f"Usuários (users)    : {users.shape[0]:,} linhas, {users.shape[1]} colunas")
    print("\n=== ESTRUTURA DE RATINGS ===")
    ratings.info(buf=out)
    print("\n=== PRIMEIRAS LINHAS DE RATINGS ===")
    print(ratings.head().to_string())
    sys.stdout = sys.__stdout__
    return out.getvalue()


def run_cell_3():
    out = io.StringIO()
    sys.stdout = out
    ratings, movies, users = load_movielens_data()
    rating_counts = ratings['rating'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['rating', 'count']
    rating_counts['percentage'] = (rating_counts['count'] / len(ratings)) * 100

    print("=== ESTATÍSTICAS DESCRITIVAS DAS NOTAS ===")
    print(f"Média: {ratings['rating'].mean():.2f}")
    print(f"Mediana: {ratings['rating'].median():.2f}")
    print(f"Desvio Padrão: {ratings['rating'].std():.2f}")
    print(f"Nota Mínima: {ratings['rating'].min()} | Nota Máxima: {ratings['rating'].max()}")
    sys.stdout = sys.__stdout__
    return out.getvalue()


def run_cell_4():
    out = io.StringIO()
    sys.stdout = out
    ratings, movies, users = load_movielens_data()
    ratings_per_user = ratings.groupby('user_id').size().reset_index(name='num_ratings')
    ratings_per_movie = ratings.groupby('item_id').size().reset_index(name='num_ratings')

    print("=== RESUMO DE AVALIAÇÕES POR USUÁRIO ===")
    print(f"Mínimo: {ratings_per_user['num_ratings'].min()} | Máximo: {ratings_per_user['num_ratings'].max()}")
    print(f"Média: {ratings_per_user['num_ratings'].mean():.1f} | Mediana: {ratings_per_user['num_ratings'].median():.1f}")
    print("\n=== RESUMO DE AVALIAÇÕES POR FILME ===")
    print(f"Mínimo: {ratings_per_movie['num_ratings'].min()} | Máximo: {ratings_per_movie['num_ratings'].max()}")
    print(f"Média: {ratings_per_movie['num_ratings'].mean():.1f} | Mediana: {ratings_per_movie['num_ratings'].median():.1f}")
    sys.stdout = sys.__stdout__
    return out.getvalue()


def run_cell_5():
    out = io.StringIO()
    sys.stdout = out
    ratings, movies, users = load_movielens_data()
    df_merged = ratings.merge(movies[['movie_id', 'movie_title']], left_on='item_id', right_on='movie_id')
    movie_stats = df_merged.groupby(['item_id', 'movie_title']).agg(
        count=('rating', 'count'),
        mean_rating=('rating', 'mean')
    ).reset_index()

    top10_populares = movie_stats.sort_values(by='count', ascending=False).head(10)
    min_ratings_threshold = 50
    top10_bem_avaliados = movie_stats[movie_stats['count'] >= min_ratings_threshold].sort_values(
        by='mean_rating', ascending=False
    ).head(10)

    print("=== TOP 10 FILMES MAIS AVALIADOS ===")
    print(top10_populares[['movie_title', 'count', 'mean_rating']].to_string(index=False))
    print(f"\n=== TOP 10 FILMES MAIS BEM AVALIADOS (MÍNIMO {min_ratings_threshold} RATINGS) ===")
    print(top10_bem_avaliados[['movie_title', 'mean_rating', 'count']].to_string(index=False))
    sys.stdout = sys.__stdout__
    return out.getvalue()


def run_cell_6():
    out = io.StringIO()
    sys.stdout = out
    ratings, movies, users = load_movielens_data()
    num_users = ratings['user_id'].nunique()
    num_movies = movies['movie_id'].nunique()
    num_ratings = len(ratings)
    total_possible_ratings = num_users * num_movies
    sparsity = 1.0 - (num_ratings / total_possible_ratings)
    density = (num_ratings / total_possible_ratings) * 100

    print("=== ANÁLISE DE ESPARSIDADE DA MATRIZ ===")
    print(f"Número de Usuários Únicos (|U|): {num_users}")
    print(f"Número de Filmes Únicos   (|I|): {num_movies}")
    print(f"Avaliações Existentes     (|R|): {num_ratings:,}")
    print(f"Avaliações Possíveis           : {total_possible_ratings:,}")
    print(f"Densidade da Matriz            : {density:.2f}%")
    print(f"Esparsidade da Matriz          : {sparsity * 100:.2f}%")
    sys.stdout = sys.__stdout__
    return out.getvalue()


def run_cell_7():
    out = io.StringIO()
    sys.stdout = out
    ratings, movies, users = load_movielens_data()
    genre_counts = movies[GENRES].sum().reset_index()
    genre_counts.columns = ['genre', 'movie_count']
    genre_counts = genre_counts.sort_values(by='movie_count', ascending=False)

    df_merged = ratings.merge(movies[['movie_id', 'movie_title'] + GENRES], left_on='item_id', right_on='movie_id')
    df_genres_ratings = df_merged[GENRES + ['rating']]
    genre_means = {}
    for g in GENRES:
        genre_means[g] = df_genres_ratings[df_genres_ratings[g] == 1]['rating'].mean()

    df_genre_means = pd.DataFrame(list(genre_means.items()), columns=['genre', 'mean_rating']).sort_values(
        by='mean_rating', ascending=False
    )

    print("=== QUANTIDADE DE FILMES POR GÊNERO ===")
    print(genre_counts.to_string(index=False))
    print("\n=== NOTA MÉDIA POR GÊNERO ===")
    print(df_genre_means.to_string(index=False))
    sys.stdout = sys.__stdout__
    return out.getvalue()


def populate_notebook():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    exec_map = {
        1: run_cell_1,
        3: run_cell_2,
        5: run_cell_3,
        8: run_cell_4,
        11: run_cell_5,
        14: run_cell_6,
        17: run_cell_7,
    }

    count = 1
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            cell["execution_count"] = count
            count += 1
            if idx in exec_map:
                txt_out = exec_map[idx]()
                cell["outputs"] = [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [line + "\n" for line in txt_out.splitlines()]
                    }
                ]

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"Notebook atualizado com saídas das células: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    populate_notebook()
