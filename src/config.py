"""
Módulo de Configuração Central do Projeto Explainable Movie Recommender.

Este arquivo centraliza caminhos de diretórios, sementes randômicas,
parâmetros do dataset e configurações do ambiente para garantir consistência
e evitar valores 'mágicos' espalhados pelo código (princípios DRY e KISS).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env (se existir) para permitir sobreposição de configurações por ambiente
load_dotenv()

# Caminho raiz do projeto (calculado dinamicamente a partir da localização deste arquivo)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Diretórios de Dados
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Diretório do Dataset MovieLens 100K
ML_100K_DIR: Path = RAW_DATA_DIR / "ml-100k"

# Outros Diretórios do Projeto
MODELS_DIR: Path = ROOT_DIR / "models"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
API_DIR: Path = ROOT_DIR / "api"
APP_DIR: Path = ROOT_DIR / "app"
DATABASE_DIR: Path = ROOT_DIR / "database"
DATABASE_PATH: Path = DATABASE_DIR / "cinematch.db"
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH.as_posix()}")

# Garantir a existência dos diretórios principais em tempo de importação
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, DATABASE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Semente Randômica Global (para reprodutibilidade dos experimentos de ML)
SEED: int = int(os.getenv("RANDOM_SEED", "42"))
RANDOM_STATE: int = SEED

# Fonte do Dataset MovieLens 100K
MOVIELENS_100K_URL: str = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Nomes de colunas padrão para o dataset MovieLens 100K (u.data, u.item, u.user)
RATINGS_COLUMNS: list[str] = ["user_id", "item_id", "rating", "timestamp"]
USER_COLUMNS: list[str] = ["user_id", "age", "gender", "occupation", "zip_code"]
GENRES: list[str] = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]
MOVIE_COLUMNS: list[str] = [
    "movie_id", "movie_title", "release_date", "video_release_date",
    "IMDb_URL"
] + GENRES
