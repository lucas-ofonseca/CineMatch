"""
Interface do Usuário (Frontend em Streamlit) — Explainable Movie Recommender.

Possui 4 Páginas:
1. Seleção de Usuário e Histórico de Avaliações
2. Recomendações em Cards com botão 'Por que?'
3. Explicação Detalhada da Recomendação com Gráfico de Influência (Plotly)
4. Dashboard Comparativo de Algoritmos (RMSE, Precision@K, Recall@K)

Design: Vanilla CSS Dark Mode / Glassmorphism elegante, responsivo e interativo.
"""

import sys
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuração da página Streamlit
st.set_page_config(
    page_title="CineMatch — Explainable Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adiciona o diretório raiz ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, API_DIR
from src.predict import get_top_n_recommendations, predict_rating
from src.explainer import explain
from database.db import SessionLocal, User as UserModel, Movie as MovieModel, Rating as RatingModel

# URL da API FastAPI
API_BASE_URL = "http://127.0.0.1:8000"


# ------------------------------------------------------------------------------
# GERENCIAMENTO DE ESTADO (st.session_state)
# ------------------------------------------------------------------------------
if "selected_user_id" not in st.session_state:
    st.session_state["selected_user_id"] = 1

if "selected_movie_id" not in st.session_state:
    st.session_state["selected_movie_id"] = 50

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "👤 Perfil e Histórico"


# ------------------------------------------------------------------------------
# INJEÇÃO DE CSS PERSONALIZADO (VANILLA CSS / GLASSMORPHISM)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Definir variáveis de tema global */
    :root {
        --bg-main: #0f0f13;
        --bg-card: rgba(26, 26, 36, 0.75);
        --accent-primary: #e50914;
        --accent-secondary: #00d2ff;
        --text-primary: #f5f5f7;
        --text-muted: #a0a0b0;
        --border-glass: rgba(255, 255, 255, 0.1);
    }

    /* Estilização Geral do Contêiner Principal */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Cartão Estilo Glassmorphism */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 210, 255, 0.3);
    }

    /* Card de Recomendação */
    .rec-card {
        background: linear-gradient(135deg, rgba(30, 30, 45, 0.9) 0%, rgba(20, 20, 30, 0.95) 100%);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Badges de Confiança */
    .badge-alta {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid #2ecc71;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-media {
        background-color: rgba(241, 196, 15, 0.2);
        color: #f1c40f;
        border: 1px solid #f1c40f;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-baixa {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid #e74c3c;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Título com Gradiente */
    .gradient-title {
        background: linear-gradient(90deg, #e50914 0%, #00d2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# FUNÇÕES DE COMUNICAÇÃO COM A API / BACKEND
# ------------------------------------------------------------------------------

def fetch_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "fallback", "model_loaded": True, "database_connected": True}


def fetch_recommendations(user_id: int, k: int = 10):
    try:
        r = requests.get(f"{API_BASE_URL}/recommendations/{user_id}?k={k}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # Fallback local direto
    return get_top_n_recommendations(user_id=user_id, n=k, include_explanations=True)


def fetch_explanation(user_id: int, movie_id: int):
    try:
        r = requests.get(f"{API_BASE_URL}/explain/{user_id}/{movie_id}", timeout=5)
        if r.status_code == 200:
            return r.json()["details"]
    except Exception:
        pass
    # Fallback local direto
    return explain(user_id=user_id, movie_id=movie_id)


def fetch_popular_movies(limit: int = 20):
    try:
        r = requests.get(f"{API_BASE_URL}/movies/popular?limit={limit}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # Fallback SQLite direto
    session = SessionLocal()
    try:
        res = (
            session.query(
                MovieModel.movie_id,
                MovieModel.movie_title.label("title"),
                MovieModel.release_date,
                MovieModel.genres_str.label("genres")
            )
            .limit(limit)
            .all()
        )
        return [{"movie_id": r.movie_id, "title": r.title, "release_date": r.release_date, "genres": r.genres} for r in res]
    finally:
        session.close()


def send_rating(user_id: int, movie_id: int, rating: float):
    try:
        r = requests.post(
            f"{API_BASE_URL}/ratings",
            json={"user_id": user_id, "movie_id": movie_id, "rating": rating},
            timeout=3
        )
        if r.status_code in [200, 201]:
            return True, r.json()["message"]
    except Exception:
        pass

    # Fallback SQLite direto
    session = SessionLocal()
    try:
        existing = session.query(RatingModel).filter_by(user_id=user_id, movie_id=movie_id).first()
        if existing:
            existing.rating = rating
        else:
            session.add(RatingModel(user_id=user_id, movie_id=movie_id, rating=rating))
        session.commit()
        return True, "Avaliação salva com sucesso (banco local)."
    except Exception as e:
        return False, str(e)
    finally:
        session.close()


# ------------------------------------------------------------------------------
# NAVEGAÇÃO LATERAL (SIDEBAR)
# ------------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/movie-beginning.png", width=70)
    st.markdown('<div class="gradient-title">CineMatch</div>', unsafe_allow_html=True)
    st.caption("Sistema Inteligente de Recomendação Explicável")

    st.markdown("---")

    # Status da API
    api_status = fetch_health()
    if api_status["status"] == "healthy":
        st.success("🟢 API FastAPI Operacional")
    else:
        st.info("🟡 Modo Autônomo / Fallback Direto")

    st.markdown("---")

    # Seleção de Páginas
    pages = [
        "👤 Perfil e Histórico",
        "🍿 Recomendações",
        "💡 Explicabilidade Detalhada",
        "📊 Dashboard de Modelos"
    ]

    selected_page = st.radio(
        "Navegação do Sistema:",
        pages,
        index=pages.index(st.session_state["current_page"])
    )
    st.session_state["current_page"] = selected_page

    st.markdown("---")

    # Seletor Global de Usuário
    st.subheader("👤 Usuário Ativo")
    user_input = st.number_input(
        "ID do Usuário (1 a 943):",
        min_value=1,
        max_value=943,
        value=int(st.session_state["selected_user_id"])
    )
    if user_input != st.session_state["selected_user_id"]:
        st.session_state["selected_user_id"] = user_input
        st.rerun()


# ==============================================================================
# PÁGINA 1: PERFIL E HISTÓRICO DE AVALIAÇÕES
# ==============================================================================
if st.session_state["current_page"] == "👤 Perfil e Histórico":
    st.markdown('<div class="gradient-title">Perfil e Histórico do Usuário</div>', unsafe_allow_html=True)
    st.write(f"Exibindo preferências e avaliações do **Usuário #{st.session_state['selected_user_id']}**.")

    session = SessionLocal()
    try:
        user_db = session.query(UserModel).filter_by(user_id=st.session_state["selected_user_id"]).first()
        ratings_query = (
            session.query(RatingModel.rating, RatingModel.timestamp, MovieModel.movie_id, MovieModel.movie_title, MovieModel.genres_str)
            .join(MovieModel, RatingModel.movie_id == MovieModel.movie_id)
            .filter(RatingModel.user_id == st.session_state["selected_user_id"])
            .order_by(RatingModel.rating.desc())
            .all()
        )
    finally:
        session.close()

    # Métricas do Perfil
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ID do Usuário", f"#{st.session_state['selected_user_id']}")
    with col2:
        st.metric("Idade", f"{user_db.age if user_db else 'N/A'} anos")
    with col3:
        st.metric("Gênero", "Masculino" if user_db and user_db.gender == 'M' else "Feminino")
    with col4:
        st.metric("Profissão", f"{user_db.occupation if user_db else 'N/A'}".capitalize())

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"📜 Histórico de Avaliações ({len(ratings_query)} filmes)")
        if ratings_query:
            data_list = []
            for r in ratings_query:
                data_list.append({
                    "ID Filme": r.movie_id,
                    "Título do Filme": r.movie_title,
                    "Nota Atribuída": f"{r.rating} ⭐",
                    "Gêneros": r.genres_str
                })
            df_hist = pd.DataFrame(data_list)
            st.dataframe(df_hist, use_container_width=True, height=400)
        else:
            st.info("Nenhuma avaliação cadastrada para este usuário.")

    with col_right:
        st.subheader("➕ Nova Avaliação")
        st.write("Avalie um filme para atualizar as recomendações:")

        popular_movies = fetch_popular_movies(limit=50)
        movie_options = {m["title"]: m["movie_id"] for m in popular_movies}

        selected_movie_title = st.selectbox("Selecione o Filme:", list(movie_options.keys()))
        new_rating = st.slider("Sua Nota:", min_value=1.0, max_value=5.0, value=4.5, step=0.5)

        if st.button("Enviar Avaliação", use_container_width=True):
            target_m_id = movie_options[selected_movie_title]
            success, msg = send_rating(st.session_state["selected_user_id"], target_m_id, new_rating)
            if success:
                st.success(f"Avaliação registrada! {msg}")
                st.rerun()
            else:
                st.error(f"Erro: {msg}")


# ==============================================================================
# PÁGINA 2: GERAR RECOMENDAÇÕES EM CARDS
# ==============================================================================
elif st.session_state["current_page"] == "🍿 Recomendações":
    st.markdown('<div class="gradient-title">Recomendações Personalizadas</div>', unsafe_allow_html=True)
    st.write(f"Filmes selecionados para o **Usuário #{st.session_state['selected_user_id']}** com base no algoritmo SVD e explicabilidade.")

    col_k, col_space = st.columns([1, 2])
    with col_k:
        k_val = st.slider("Quantidade de Recomendações (Top-K):", min_value=1, max_value=20, value=6)

    with st.spinner("Gerando recomendações e calculando evidências explicativas..."):
        recs = fetch_recommendations(st.session_state["selected_user_id"], k=k_val)

    if recs:
        # Exibir em grid de 3 colunas
        cols_per_row = 3
        for i in range(0, len(recs), cols_per_row):
            row_recs = recs[i:i + cols_per_row]
            cols = st.columns(cols_per_row)

            for idx, item in enumerate(row_recs):
                with cols[idx]:
                    exp_data = item.get("explanation", {})
                    conf_level = exp_data.get("confidence", {}).get("level", "Média")
                    badge_class = f"badge-{conf_level.lower()}"

                    st.markdown(f"""
                    <div class="rec-card">
                        <div>
                            <span class="{badge_class}">Confiança {conf_level}</span>
                            <h4 style="margin-top: 10px; color: #fff;">{item['title']}</h4>
                            <p style="font-size: 1.2rem; font-weight: 700; color: #00d2ff;">Nota Prevista: {item['predicted_rating']} / 5.0 ⭐</p>
                            <p style="font-size: 0.9rem; color: #a0a0b0;">{exp_data.get('explanation_text', 'Sem detalhes')[:140]}...</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Botão 'Por que?' para navegar à Página 3
                    if st.button(f"Por que? #{item['movie_id']}", key=f"btn_why_{item['movie_id']}", use_container_width=True):
                        st.session_state["selected_movie_id"] = item["movie_id"]
                        st.session_state["current_page"] = "💡 Explicabilidade Detalhada"
                        st.rerun()


# ==============================================================================
# PÁGINA 3: EXPLICAÇÃO DETALHADA DA RECOMENDAÇÃO
# ==============================================================================
elif st.session_state["current_page"] == "💡 Explicabilidade Detalhada":
    st.markdown('<div class="gradient-title">Detalhamento da Explicabilidade</div>', unsafe_allow_html=True)

    col_m, col_u = st.columns([2, 1])
    with col_m:
        popular_movies = fetch_popular_movies(limit=100)
        movie_dict = {m["title"]: m["movie_id"] for m in popular_movies}

        # Tentar selecionar o filme do session_state
        selected_m_id = st.number_input(
            "ID do Filme Alvo:",
            min_value=1,
            max_value=1682,
            value=int(st.session_state["selected_movie_id"])
        )
        st.session_state["selected_movie_id"] = selected_m_id

    exp = fetch_explanation(st.session_state["selected_user_id"], st.session_state["selected_movie_id"])

    if exp:
        # Card principal de destaque
        st.markdown(f"""
        <div class="glass-card">
            <h2>🎬 {exp['movie_title']}</h2>
            <p style="font-size: 1.3rem; color: #00d2ff; font-weight: 700;">Nota Prevista pelo SVD: {exp['predicted_rating']} / 5.0 ⭐</p>
            <p style="font-size: 1.1rem; line-height: 1.6; color: #f5f5f7;">{exp['explanation_text']}</p>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            # Gráfico de Influência Plotly
            st.subheader("🎯 Filmes do seu histórico que mais influenciaram")
            influential = exp.get("influential_history_movies", [])
            if influential:
                df_inf = pd.DataFrame(influential)
                fig_inf = px.bar(
                    df_inf,
                    x="similarity",
                    y="title",
                    orientation="h",
                    text=df_inf["similarity"].apply(lambda x: f"{int(x*100)}% afinidade"),
                    title="<b>Afinidade Temática nos Vetores Latentes (SVD)</b>",
                    labels={"similarity": "Similaridade de Cosseno", "title": "Filme do seu Histórico"},
                    color="similarity",
                    color_continuous_scale="Blues"
                )
                fig_inf.update_layout(yaxis={"categoryorder": "total ascending"}, height=350)
                st.plotly_chart(fig_inf, use_container_width=True)
            else:
                st.info("Nenhum filme com nota >= 4.0 encontrado no seu histórico para comparação de vetores.")

        with col_right:
            # Perfil de Gêneros do Usuário
            st.subheader("📊 Seus Gêneros Preferidos")
            top_genres = exp.get("user_top_genres", [])
            if top_genres:
                df_gen = pd.DataFrame(top_genres)
                fig_gen = px.pie(
                    df_gen,
                    names="genre",
                    values="count",
                    title="<b>Distribuição de Gêneros com Nota >= 4.0</b>",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_gen.update_layout(height=350)
                st.plotly_chart(fig_gen, use_container_width=True)
            else:
                st.info("Sem histórico suficiente de gêneros para este usuário.")

        # Destaque de Usuários Similares
        st.subheader("👥 Espectadores com Gosto Similar ao Seu")
        similar_users = exp.get("similar_users_highlights", [])
        if similar_users:
            cols_u = st.columns(len(similar_users))
            for i, u in enumerate(similar_users):
                with cols_u[i]:
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 15px; text-align: center;">
                        <h4>{u['anonymous_user_label']}</h4>
                        <p style="color: #00d2ff; font-weight: 700;">Afinidade: {u['similarity_score']}%</p>
                        <p>Avaliou este filme com nota <b>{u['rating_given']} ⭐</b></p>
                    </div>
                    """, unsafe_allow_html=True)


# ==============================================================================
# PÁGINA 4: DASHBOARD COMPARATIVO DE ALGORITMOS
# ==============================================================================
elif st.session_state["current_page"] == "📊 Dashboard de Modelos":
    st.markdown('<div class="gradient-title">Dashboard Comparativo de Algoritmos</div>', unsafe_allow_html=True)
    st.write("Benchmarking completo dos modelos de recomendação em 5-Fold Cross Validation.")

    comp_file = MODELS_DIR / "model_comparison.csv"
    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)

        # Destacar o campeão
        champion = df_comp.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Modelo Campeão", "SVD (100 fatores)")
        with col2:
            st.metric("Menor RMSE", f"{champion['rmse']:.4f}")
        with col3:
            st.metric("Maior Precision@10", f"{champion['precision_at_10']:.4f}")
        with col4:
            st.metric("Maior Recall@10", f"{champion['recall_at_10']:.4f}")

        st.markdown("---")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_rmse = px.bar(
                df_comp,
                x="rmse",
                y="model_name",
                orientation="h",
                text="rmse",
                title="<b>Comparativo de Erro Quadrático Médio (RMSE) — Menor é Melhor</b>",
                color="rmse",
                color_continuous_scale="Reds_r"
            )
            fig_rmse.update_layout(yaxis={"categoryorder": "total descending"}, height=450)
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col_g2:
            fig_prec = px.bar(
                df_comp,
                x="precision_at_10",
                y="model_name",
                orientation="h",
                text="precision_at_10",
                title="<b>Comparativo de Precisão (Precision@10) — Maior é Melhor</b>",
                color="precision_at_10",
                color_continuous_scale="Viridis"
            )
            fig_prec.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
            st.plotly_chart(fig_prec, use_container_width=True)

        st.subheader("📋 Tabela Completa de Métricas")
        st.dataframe(df_comp, use_container_width=True)

    else:
        st.warning("Arquivo de comparação de modelos não encontrado. Treine os modelos via `python src/train.py` primeiro.")
