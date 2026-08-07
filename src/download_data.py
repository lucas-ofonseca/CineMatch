"""
Script de Download e Extração do Dataset MovieLens 100K.

Utiliza urllib e zipfile para baixar autonomamente o dataset
e descompactar no diretório data/raw configurado em src/config.py.
"""

import sys
import zipfile
import urllib.request
from pathlib import Path

# Adiciona o diretório raiz ao sys.path para permitir importação de src.config
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import RAW_DATA_DIR, ML_100K_DIR, MOVIELENS_100K_URL


def download_and_extract_movielens() -> None:
    """
    Baixa o arquivo zip do MovieLens 100K e o extrai em data/raw/ml-100k.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DATA_DIR / "ml-100k.zip"

    # Verifica se os dados já foram extraídos anteriormente
    if (ML_100K_DIR / "u.data").exists():
        print(f"Dataset MovieLens 100K já se encontra em: {ML_100K_DIR}")
        return

    print(f"Fazendo download do MovieLens 100K a partir de: {MOVIELENS_100K_URL}")
    try:
        urllib.request.urlretrieve(MOVIELENS_100K_URL, zip_path)
        print(f"Download concluído com sucesso. Salvado em: {zip_path}")
    except Exception as e:
        print(f"Erro ao realizar download do dataset: {e}")
        sys.exit(1)

    print(f"Extraindo {zip_path} em {RAW_DATA_DIR}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print(f"Extração concluída com sucesso! Dados extraídos em: {ML_100K_DIR}")
    except Exception as e:
        print(f"Erro ao extrair o arquivo zip: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_and_extract_movielens()
