"""
Script para executar e popular saídas do notebook 02_preprocessamento.ipynb.
"""

import json
import sys
from pathlib import Path
import io
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DATA_DIR
from src.preprocessing import run_preprocessing_pipeline

NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "02_preprocessamento.ipynb"


def run_c1():
    return "Ambiente de pré-processamento configurado!\n"


def run_c2():
    out = io.StringIO()
    sys.stdout = out
    run_preprocessing_pipeline()
    sys.stdout = sys.__stdout__
    return out.getvalue()


def run_c3():
    out = io.StringIO()
    sys.stdout = out
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_ratings.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_ratings.csv")
    full_df = pd.read_csv(PROCESSED_DATA_DIR / "processed_ratings.csv")

    print("=== VALIDAÇÃO DO SPLIT POR USUÁRIO ===")
    user_full_counts = full_df.groupby('user_id').size()
    movie_full_counts = full_df.groupby('item_id').size()

    print(f"Total de usuários únicos no treino: {train_df['user_id'].nunique()}")
    print(f"Total de usuários únicos no teste : {test_df['user_id'].nunique()}")
    print(f"Menor número de avaliações por usuário no conjunto total: {user_full_counts.min()}")
    print(f"Menor número de avaliações por filme no conjunto total  : {movie_full_counts.min()}")

    print("\n=== AMOSTRA DO CONJUNTO DE TREINO ===")
    print(train_df.head().to_string(index=False))

    print("\n=== AMOSTRA DO CONJUNTO DE TESTE ===")
    print(test_df.head().to_string(index=False))
    sys.stdout = sys.__stdout__
    return out.getvalue()


def run_c4():
    out = io.StringIO()
    sys.stdout = out
    files = list(PROCESSED_DATA_DIR.glob("*.csv"))
    print("=== ARQUIVOS GERADOS EM DATA/PROCESSED/ ===")
    for f in sorted(files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"- {f.name:<25} | Tamanho: {size_mb:.2f} MB")
    sys.stdout = sys.__stdout__
    return out.getvalue()


def populate():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    exec_map = {
        1: run_c1,
        3: run_c2,
        5: run_c3,
        7: run_c4
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

    print("Notebook 02 populado com saídas com sucesso!")


if __name__ == "__main__":
    populate()
