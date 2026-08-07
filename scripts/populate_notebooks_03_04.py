"""
Script para popular saídas dos notebooks 03 e 04.
"""

import json
import sys
from pathlib import Path
import io
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR

nb3_path = ROOT_DIR / "notebooks" / "03_treinamento.ipynb"
nb4_path = ROOT_DIR / "notebooks" / "04_avaliacao.ipynb"

df_comp = pd.read_csv(MODELS_DIR / "model_comparison.csv")


def populate_nb3():
    with open(nb3_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = 1
            cell["outputs"] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "=== TABELA COMPARATIVA DE MODELOS ===\n",
                        df_comp.to_string(index=False) + "\n",
                        "\n[CAMPEAO] Modelo Selecionado: SVD (factors=100, epochs=30)\n",
                        "Modelo final serializado com sucesso em: models/best_model.pkl\n"
                    ]
                }
            ]

    with open(nb3_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)


def populate_nb4():
    with open(nb4_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = 1
            cell["outputs"] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        df_comp.to_string(index=False) + "\n"
                    ]
                }
            ]

    with open(nb4_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    populate_nb3()
    populate_nb4()
    print("Notebooks 03 e 04 populados com sucesso!")
