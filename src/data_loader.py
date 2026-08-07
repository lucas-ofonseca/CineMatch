"""
Módulo de Carregamento de Dados do MovieLens.

Fornece funções utilitárias para carregar e estruturar os DataFrames de 
avaliações (ratings), filmes (movies) e usuários (users) a partir dos arquivos brutos
do MovieLens 100K (u.data, u.item, u.user).
"""

import sys
from pathlib import Path
import pandas as pd

# Garante acesso ao pacote src independente do diretório de execução
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ML_100K_DIR,
    RATINGS_COLUMNS,
    USER_COLUMNS,
    MOVIE_COLUMNS,
    GENRES
)


def load_ratings() -> pd.DataFrame:
    """
    Carrega o arquivo u.data (avaliações dos usuários).
    Separador: tabulação ('\t').
    """
    ratings_path = ML_100K_DIR / "u.data"
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Arquivo de avaliações não encontrado em: {ratings_path}. "
            "Execute 'python src/download_data.py' primeiro."
        )

    # Converter timestamp para datetime para facilitar análises temporais futuras
    df_ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=RATINGS_COLUMNS,
        engine="python"
    )
    df_ratings["datetime"] = pd.to_datetime(df_ratings["timestamp"], unit="s")
    return df_ratings


def load_movies() -> pd.DataFrame:
    """
    Carrega o arquivo u.item (informações e gêneros dos filmes).
    Separador: pipe ('|'). Encoding: latin-1 (necessário devido a acentos/caracteres especiais nos títulos).
    """
    movies_path = ML_100K_DIR / "u.item"
    if not movies_path.exists():
        raise FileNotFoundError(
            f"Arquivo de filmes não encontrado em: {movies_path}. "
            "Execute 'python src/download_data.py' primeiro."
        )

    df_movies = pd.read_csv(
        movies_path,
        sep="|",
        names=MOVIE_COLUMNS,
        encoding="latin-1",
        engine="python"
    )
    return df_movies


def load_users() -> pd.DataFrame:
    """
    Carrega o arquivo u.user (dados demográficos dos usuários).
    Separador: pipe ('|').
    """
    users_path = ML_100K_DIR / "u.user"
    if not users_path.exists():
        raise FileNotFoundError(
            f"Arquivo de usuários não encontrado em: {users_path}. "
            "Execute 'python src/download_data.py' primeiro."
        )

    df_users = pd.read_csv(
        users_path,
        sep="|",
        names=USER_COLUMNS,
        encoding="latin-1",
        engine="python"
    )
    return df_users


def load_movielens_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega todos os 3 DataFrames do MovieLens 100K.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (ratings, movies, users)
    """
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()
    return ratings, movies, users


if __name__ == "__main__":
    r, m, u = load_movielens_data()
    print(f"Ratings shape: {r.shape}")
    print(f"Movies shape: {m.shape}")
    print(f"Users shape: {u.shape}")
