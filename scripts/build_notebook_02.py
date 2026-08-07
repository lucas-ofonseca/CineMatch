"""
Script para geração e execução do notebook 02_preprocessamento.ipynb.
"""

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "02_preprocessamento.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ⚙️ Notebook 02: Pré-processamento e Preparação dos Dados\n",
            "**Projeto**: Explainable Movie Recommender  \n",
            "**Modulo**: `src/preprocessing.py`  \n",
            "\n",
            "---\n",
            "\n",
            "## 🎯 Objetivos deste Notebook\n",
            "1. **Limpeza e Tratamento de Nulos**: Tratar dados ausentes e garantir tipos numéricos consistentes.\n",
            "2. **Filtragem Iterativa k-Core**: Garantir que cada usuário possua no mínimo 20 avaliações e cada filme tenha no mínimo 10 avaliações.\n",
            "3. **Split Treino/Teste (80/20) Estratificado por Usuário**: Garantir a distribuição proporcional de treino e teste para cada usuário individualmente.\n",
            "4. **Persistência**: Verificar os arquivos serializados em `data/processed/`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "ROOT_DIR = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n",
            "if str(ROOT_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(ROOT_DIR))\n",
            "\n",
            "from src.config import PROCESSED_DATA_DIR, SEED\n",
            "from src.preprocessing import run_preprocessing_pipeline\n",
            "\n",
            "print(\"Ambiente de pré-processamento configurado!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Execução da Pipeline de Pré-processamento"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_df, test_df, movies_df, users_df = run_preprocessing_pipeline()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Validação da Integridade do Split e dos Limiares"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=== VALIDAÇÃO DO SPLIT POR USUÁRIO ===\")\n",
            "full_df = pd.concat([train_df, test_df])\n",
            "user_train_counts = train_df.groupby('user_id').size()\n",
            "user_test_counts = test_df.groupby('user_id').size()\n",
            "user_full_counts = full_df.groupby('user_id').size()\n",
            "\n",
            "print(f\"Total de usuários únicos no treino: {train_df['user_id'].nunique()}\")\n",
            "print(f\"Total de usuários únicos no teste : {test_df['user_id'].nunique()}\")\n",
            "print(f\"Menor número de avaliações por usuário no conjunto total: {user_full_counts.min()}\")\n",
            "\n",
            "movie_full_counts = full_df.groupby('item_id').size()\n",
            "print(f\"Menor número de avaliações por filme no conjunto total  : {movie_full_counts.min()}\")\n",
            "\n",
            "print(\"\\n=== AMOSTRA DO CONJUNTO DE TREINO ===\")\n",
            "print(train_df.head())\n",
            "\n",
            "print(\"\\n=== AMOSTRA DO CONJUNTO DE TESTE ===\")\n",
            "print(test_df.head())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Inspeção dos Arquivos Gerados em `data/processed/`"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "files = list(PROCESSED_DATA_DIR.glob(\"*.csv\"))\n",
            "print(\"=== ARQUIVOS GERADOS EM DATA/PROCESSED/ ===\")\n",
            "for f in sorted(files):\n",
            "    size_mb = f.stat().st_size / (1024 * 1024)\n",
            "    print(f\"- {f.name:<25} | Tamanho: {size_mb:.2f} MB\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Conclusão da Etapa 2\n",
            "\n",
            "O conjunto de dados está devidamente limpo, filtrado com o algoritmo *k-core*, dividido em Treino/Teste estratificado por usuário e persistido em `data/processed/`.\n",
            "\n",
            "---\n",
            "**Próxima Etapa no Roadmap**: **Etapa 3 — Treinamento dos Modelos de Recomendação** (`src/train.py` e `notebooks/03_treinamento.ipynb`)."
        ]
    }
]

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2, ensure_ascii=False)

print(f"Notebook gerado com sucesso em: {NOTEBOOK_PATH}")
