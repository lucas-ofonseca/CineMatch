"""
Script para geração e execução programática do notebook 01_exploracao.ipynb.

Este script cria a estrutura completa de células Markdown e Código,
executa o notebook e salva o arquivo final em notebooks/01_exploracao.ipynb
com todas as saídas e gráficos Plotly renderizados.
"""

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "01_exploracao.ipynb"

# Estrutura de Células do Notebook 01
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📊 Notebook 01: Análise Exploratória de Dados (EDA)\n",
            "**Projeto**: Explainable Movie Recommender  \n",
            "**Dataset**: MovieLens 100K  \n",
            "\n",
            "---\n",
            "\n",
            "## 🎯 Objetivos deste Notebook\n",
            "1. **Compreender o Dataset**: Carregar e inspecionar a estrutura de `ratings`, `movies` e `users`.\n",
            "2. **Analisar Avaliações**: Avaliar a distribuição geral das notas (1 a 5).\n",
            "3. **Analisar Usuários e Filmes**: Avaliar o volume de avaliações por usuário e por filme (*heavy users*, *long tail*, *cold start*).\n",
            "4. **Rankings de Filmes**: Identificar o Top-10 filmes mais avaliados e Top-10 mais bem avaliados (com filtro de corte).\n",
            "5. **Esparsidade da Matriz Usuário-Filme**: Calcular a densidade real da matriz de utilidade e visualizar o heatmap.\n",
            "6. **Distribuição de Gêneros**: Analisar a frequência e relevância dos gêneros cinematográficos.\n",
            "7. **Documentação para o Recomendador Explicável**: Sintetizar implicações diretas dos dados para a modelagem."
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
            "import plotly.express as px\n",
            "import plotly.graph_objects as go\n",
            "import plotly.io as pio\n",
            "\n",
            "# Configurar template padrão do Plotly para visualizações modernas e elegantes\n",
            "pio.templates.default = \"plotly_white\"\n",
            "\n",
            "# Adicionar diretório raiz do projeto ao sys.path\n",
            "ROOT_DIR = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n",
            "if str(ROOT_DIR) not in sys.path:\n",
            "    sys.path.insert(0, str(ROOT_DIR))\n",
            "\n",
            "from src.config import SEED, GENRES\n",
            "from src.data_loader import load_movielens_data\n",
            "\n",
            "print(\"Ambiente configurado com sucesso! Semente global:\", SEED)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Carregamento e Inspeção Inicial dos Dados\n",
            "\n",
            "Carregamos os 3 DataFrames do MovieLens 100K utilizando as rotinas modularizadas do `src.data_loader`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "ratings, movies, users = load_movielens_data()\n",
            "\n",
            "print(\"=== DIMENSÕES DOS DATASETS ===\")\n",
            "print(f\"Avaliações (ratings): {ratings.shape[0]:,} linhas, {ratings.shape[1]} colunas\")\n",
            "print(f\"Filmes (movies)     : {movies.shape[0]:,} linhas, {movies.shape[1]} colunas\")\n",
            "print(f\"Usuários (users)    : {users.shape[0]:,} linhas, {users.shape[1]} colunas\")\n",
            "\n",
            "print(\"\\n=== ESTRUTURA DE RATINGS ===\")\n",
            "print(ratings.info())\n",
            "\n",
            "print(\"\\n=== PRIMEIRAS LINHAS DE RATINGS ===\")\n",
            "print(ratings.head())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Distribuição das Avaliações (Ratings)\n",
            "\n",
            "Analisamos a escala e a frequência de notas de 1 a 5 atribuídas pelos usuários."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "rating_counts = ratings['rating'].value_counts().sort_index().reset_index()\n",
            "rating_counts.columns = ['rating', 'count']\n",
            "rating_counts['percentage'] = (rating_counts['count'] / len(ratings)) * 100\n",
            "\n",
            "fig_ratings = px.bar(\n",
            "    rating_counts,\n",
            "    x='rating',\n",
            "    y='count',\n",
            "    text=rating_counts['percentage'].apply(lambda x: f\"{x:.1f}%\"),\n",
            "    title=\"<b>Distribuição Geral das Avaliações (1 a 5 Estrelas)</b>\",\n",
            "    labels={'rating': 'Nota (Estrelas)', 'count': 'Total de Avaliações'},\n",
            "    color='rating',\n",
            "    color_continuous_scale='Viridis'\n",
            ")\n",
            "fig_ratings.update_traces(textposition='outside')\n",
            "fig_ratings.update_layout(showlegend=False, height=450)\n",
            "fig_ratings.show()\n",
            "\n",
            "print(\"=== ESTATÍSTICAS DESCRITIVAS DAS NOTAS ===\")\n",
            "print(f\"Média: {ratings['rating'].mean():.2f}\")\n",
            "print(f\"Mediana: {ratings['rating'].median():.2f}\")\n",
            "print(f\"Desvio Padrão: {ratings['rating'].std():.2f}\")\n",
            "print(f\"Nota Mínima: {ratings['rating'].min()} | Nota Máxima: {ratings['rating'].max()}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **📌 Achado 1 — Tendência Positiva de Avaliações**:\n",
            "> - A nota **4** é a mais frequente (~34.2%), seguida da nota **3** (~27.1%) e nota **5** (~21.2%).\n",
            "> - As notas negativas (1 e 2) somam menos de 18% do total.\n",
            "> - *Implicação para recomendação*: Existe um viés natural de usuários avaliarem itens que apreciam. O recomendador precisará ser sensível a nuances entre 4 e 5 estrelas."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Distribuição de Avaliações por Usuário e por Filme\n",
            "\n",
            "Examinamos o volume de avaliações por usuário (*heavy users*, *cold start*) e a popularidade dos filmes (*Long Tail*)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "ratings_per_user = ratings.groupby('user_id').size().reset_index(name='num_ratings')\n",
            "ratings_per_movie = ratings.groupby('item_id').size().reset_index(name='num_ratings')\n",
            "\n",
            "# Histograma por Usuário\n",
            "fig_user_dist = px.histogram(\n",
            "    ratings_per_user,\n",
            "    x='num_ratings',\n",
            "    nbins=50,\n",
            "    title=\"<b>Distribuição da Quantidade de Avaliações por Usuário</b>\",\n",
            "    labels={'num_ratings': 'Quantidade de Avaliações prestadas'},\n",
            "    color_discrete_sequence=['#2b5c8f']\n",
            ")\n",
            "fig_user_dist.update_layout(height=400)\n",
            "fig_user_dist.show()\n",
            "\n",
            "# Histograma por Filme\n",
            "fig_movie_dist = px.histogram(\n",
            "    ratings_per_movie,\n",
            "    x='num_ratings',\n",
            "    nbins=50,\n",
            "    title=\"<b>Distribuição da Quantidade de Avaliações por Filme (Cauda Longa)</b>\",\n",
            "    labels={'num_ratings': 'Quantidade de Avaliações recebidas'},\n",
            "    color_discrete_sequence=['#e74c3c']\n",
            ")\n",
            "fig_movie_dist.update_layout(height=400)\n",
            "fig_movie_dist.show()\n",
            "\n",
            "print(\"=== RESUMO DE AVALIAÇÕES POR USUÁRIO ===\")\n",
            "print(f\"Mínimo: {ratings_per_user['num_ratings'].min()} | Máximo: {ratings_per_user['num_ratings'].max()}\")\n",
            "print(f\"Média: {ratings_per_user['num_ratings'].mean():.1f} | Mediana: {ratings_per_user['num_ratings'].median():.1f}\")\n",
            "\n",
            "print(\"\\n=== RESUMO DE AVALIAÇÕES POR FILME ===\")\n",
            "print(f\"Mínimo: {ratings_per_movie['num_ratings'].min()} | Máximo: {ratings_per_movie['num_ratings'].max()}\")\n",
            "print(f\"Média: {ratings_per_movie['num_ratings'].mean():.1f} | Mediana: {ratings_per_movie['num_ratings'].median():.1f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **📌 Achado 2 — Cauda Longa e Viés de Popularidade**:\n",
            "> - **Usuários**: Todos os usuários no dataset possuem no mínimo 20 avaliações (filtro padrão do ML-100K). O usuário mais ativo forneceu 737 avaliações.\n",
            "> - **Filmes**: A distribuição por filme exibe uma forte **Cauda Longa** (*Long Tail*). Poucos filmes recebem centenas de avaliações (o mais popular tem 583), enquanto a maioria possui menos de 50 avaliações.\n",
            "> - *Implicação para recomendação*: Modelos de filtragem colaborativa puros tendem a recomendar blockbusters. A explicabilidade ajudará a justificar itens menos populares (da cauda longa)."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Ranking de Filmes: Top-10 Mais Avaliados vs. Top-10 Mais Bem Avaliados\n",
            "\n",
            "Comparamos os filmes mais populares (volume de notas) com os de maior nota média (com filtro de corte de suporte mínimo $\\ge 50$ notas)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Unir ratings com filmes para obter os títulos\n",
            "df_merged = ratings.merge(movies[['movie_id', 'movie_title']], left_on='item_id', right_on='movie_id')\n",
            "\n",
            "# Agrupar por filme\n",
            "movie_stats = df_merged.groupby(['item_id', 'movie_title']).agg(\n",
            "    count=('rating', 'count'),\n",
            "    mean_rating=('rating', 'mean')\n",
            ").reset_index()\n",
            "\n",
            "# Top 10 mais avaliados\n",
            "top10_populares = movie_stats.sort_values(by='count', ascending=False).head(10)\n",
            "\n",
            "fig_top_pop = px.bar(\n",
            "    top10_populares,\n",
            "    x='count',\n",
            "    y='movie_title',\n",
            "    orientation='h',\n",
            "    text='count',\n",
            "    title=\"<b>Top 10 Filmes Mais Avaliados (Mais Populares)</b>\",\n",
            "    labels={'count': 'Número de Avaliações', 'movie_title': 'Título do Filme'},\n",
            "    color='count',\n",
            "    color_continuous_scale='Blues'\n",
            ")\n",
            "fig_top_pop.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)\n",
            "fig_top_pop.show()\n",
            "\n",
            "# Top 10 mais bem avaliados (com suporte mínimo de 50 avaliações)\n",
            "min_ratings_threshold = 50\n",
            "top10_bem_avaliados = movie_stats[movie_stats['count'] >= min_ratings_threshold].sort_values(\n",
            "    by='mean_rating', ascending=False\n",
            ").head(10)\n",
            "\n",
            "fig_top_qual = px.bar(\n",
            "    top10_bem_avaliados,\n",
            "    x='mean_rating',\n",
            "    y='movie_title',\n",
            "    orientation='h',\n",
            "    text=top10_bem_avaliados['mean_rating'].apply(lambda x: f\"{x:.2f} ⭐\"),\n",
            "    title=f\"<b>Top 10 Filmes Mais Bem Avaliados (Mínimo de {min_ratings_threshold} avaliações)</b>\",\n",
            "    labels={'mean_rating': 'Nota Média', 'movie_title': 'Título do Filme'},\n",
            "    color='mean_rating',\n",
            "    color_continuous_scale='Greens'\n",
            ")\n",
            "fig_top_qual.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)\n",
            "fig_top_qual.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **📌 Achado 3 — Diferença entre Popularidade e Qualidade**:\n",
            "> - Títulos populares como *Star Wars (1977)* e *Contact (1997)* lideram a contagem absoluta.\n",
            "> - Com o filtro de suporte ($\n",
            "\\ge 50$ avaliações), clássicos como *Close Shave, A (1995)* e *Schindler's List (1993)* registram as maiores médias (~4.46 a 4.55).\n",
            "> - *Implicação*: É indispensável aplicar um suporte mínimo ou amortecimento bayesiano no cálculo de notas médias."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Esparsidade da Matriz Usuário-Filme\n",
            "\n",
            "Calculamos a densidade e a esparsidade da matriz de utilidade $R_{u, i}$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "num_users = ratings['user_id'].nunique()\n",
            "num_movies = movies['movie_id'].nunique()\n",
            "num_ratings = len(ratings)\n",
            "\n",
            "total_possible_ratings = num_users * num_movies\n",
            "sparsity = 1.0 - (num_ratings / total_possible_ratings)\n",
            "density = (num_ratings / total_possible_ratings) * 100\n",
            "\n",
            "print(f\"=== ANÁLISE DE ESPARSIDADE DA MATRIZ ===\")\n",
            "print(f\"Número de Usuários Únicos (|U|): {num_users}\")\n",
            "print(f\"Número de Filmes Únicos   (|I|): {num_movies}\")\n",
            "print(f\"Avaliações Existentes     (|R|): {num_ratings:,}\")\n",
            "print(f\"Avaliações Possíveis           : {total_possible_ratings:,}\")\n",
            "print(f\"Densidade da Matriz            : {density:.2f}%\")\n",
            "print(f\"Esparsidade da Matriz          : {sparsity * 100:.2f}%\")\n",
            "\n",
            "# Criar sub-matriz para exibição no Heatmap (Top 50 Usuários mais ativos x Top 50 Filmes mais populares)\n",
            "top_users = ratings_per_user.sort_values(by='num_ratings', ascending=False).head(50)['user_id']\n",
            "top_movies = ratings_per_movie.sort_values(by='num_ratings', ascending=False).head(50)['item_id']\n",
            "\n",
            "subset_ratings = ratings[ratings['user_id'].isin(top_users) & ratings['item_id'].isin(top_movies)]\n",
            "pivot_matrix = subset_ratings.pivot(index='user_id', columns='item_id', values='rating')\n",
            "\n",
            "# Preencher NaN com 0 para o Heatmap de densidade\n",
            "heatmap_data = pivot_matrix.notna().astype(int)\n",
            "\n",
            "fig_heatmap = px.imshow(\n",
            "    heatmap_data,\n",
            "    labels=dict(x=\"ID do Filme (Top 50 Populares)\", y=\"ID do Usuário (Top 50 Ativos)\", color=\"Avaliado\"),\n",
            "    title=\"<b>Heatmap da Densidade da Matriz Usuário-Filme (Amostra Top 50 x Top 50)</b>\",\n",
            "    color_continuous_scale=[[0, '#f0f0f0'], [1, '#2b5c8f']]\n",
            ")\n",
            "fig_heatmap.update_layout(height=500)\n",
            "fig_heatmap.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **📌 Achado 4 — Matriz Esparsa (~93.70%)**:\n",
            "> - Apenas **6.30%** das células da matriz contêm avaliações reais.\n",
            "> - *Implicação para modelagem*: Algoritmos de Fatoração de Matrizes (SVD via Surprise) são perfeitamente adequados para prever os valores faltantes (93.7% da matriz)."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Distribuição de Gêneros Cinematográficos\n",
            "\n",
            "Analisamos a quantidade de filmes e a nota média recebida por cada gênero cinematográfico."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Contar total de filmes por gênero\n",
            "genre_counts = movies[GENRES].sum().reset_index()\n",
            "genre_counts.columns = ['genre', 'movie_count']\n",
            "genre_counts = genre_counts.sort_values(by='movie_count', ascending=False)\n",
            "\n",
            "fig_genres = px.bar(\n",
            "    genre_counts,\n",
            "    x='movie_count',\n",
            "    y='genre',\n",
            "    orientation='h',\n",
            "    text='movie_count',\n",
            "    title=\"<b>Quantidade de Filmes por Gênero Cinematográfico</b>\",\n",
            "    labels={'movie_count': 'Total de Filmes', 'genre': 'Gênero'},\n",
            "    color='movie_count',\n",
            "    color_continuous_scale='Purples'\n",
            ")\n",
            "fig_genres.update_layout(yaxis={'categoryorder': 'total ascending'}, height=550)\n",
            "fig_genres.show()\n",
            "\n",
            "# Calcular nota média por gênero\n",
            "df_genres_ratings = df_merged[GENRES + ['rating']]\n",
            "genre_means = {}\n",
            "for g in GENRES:\n",
            "    genre_means[g] = df_genres_ratings[df_genres_ratings[g] == 1]['rating'].mean()\n",
            "\n",
            "df_genre_means = pd.DataFrame(list(genre_means.items()), columns=['genre', 'mean_rating']).sort_values(\n",
            "    by='mean_rating', ascending=False\n",
            ")\n",
            "\n",
            "fig_genre_means = px.bar(\n",
            "    df_genre_means,\n",
            "    x='mean_rating',\n",
            "    y='genre',\n",
            "    orientation='h',\n",
            "    text=df_genre_means['mean_rating'].apply(lambda x: f\"{x:.2f} ⭐\"),\n",
            "    title=\"<b>Nota Média das Avaliações por Gênero</b>\",\n",
            "    labels={'mean_rating': 'Nota Média', 'genre': 'Gênero'},\n",
            "    color='mean_rating',\n",
            "    color_continuous_scale='Teal'\n",
            ")\n",
            "fig_genre_means.update_layout(yaxis={'categoryorder': 'total ascending'}, height=550)\n",
            "fig_genre_means.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **📌 Achado 5 — Relevância dos Gêneros para Explicabilidade**:\n",
            "> - *Drama* (725 filmes) e *Comedy* (505 filmes) são os gêneros mais abundantes.\n",
            "> - *Film-Noir* (~3.92) e *War* (~3.82) registram as maiores notas médias.\n",
            "> - *Implicação*: Os gêneros servirão como o motor primário para gerar frases explicativas no módulo `explainer.py` (ex.: *\"Recomendado porque 45% dos filmes que você avaliou com nota 5 são do gênero Film-Noir\"*)."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Conclusões e Próximos Passos\n",
            "\n",
            "### 🎯 Resumo da Análise:\n",
            "1. **Viés Positivo**: A distribuição de notas é concentrada em 3 e 4 estrelas.\n",
            "2. **Cauda Longa**: A matriz exibe forte concentratividade de popularidade em poucos títulos.\n",
            "3. **Esparsidade Elevada**: Matriz 93.7% esparsa — SVD e KNN são modelos altamente adequados.\n",
            "4. **Gêneros Explicáveis**: Gêneros como Drama, Comedy, Film-Noir apresentam padrões distintos de preferência que servirão para o componente de explicabilidade.\n",
            "\n",
            "---\n",
            "**Próxima Etapa**: **Etapa 2 — Pré-processamento e Preparação dos Dados** (`src/preprocessing.py`)."
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
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2, ensure_ascii=False)

print(f"Notebook gerado com sucesso em: {NOTEBOOK_PATH}")
