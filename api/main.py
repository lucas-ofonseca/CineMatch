"""
API RESTful do Sistema Explainable Movie Recommender (FastAPI).

Endpoints fornecidos:
1. GET /health -> Verificação de integridade da API, modelo e banco de dados.
2. GET /recommendations/{user_id}?k=10 -> Gera o Top-K de recomendações explicáveis.
3. GET /explain/{user_id}/{movie_id} -> Retorna o detalhamento da explicação para um filme.
4. GET /movies/popular?limit=20 -> Retorna o catálogo dos filmes mais populares/bem avaliados.
5. POST /ratings -> Registra uma nova avaliação de usuário no banco de dados.

Suporta CORS para consumo via Streamlit e documentação via Swagger UI.
"""

import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Adiciona o diretório raiz do projeto ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import func

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.predict import predict_rating, get_top_n_recommendations
from src.explainer import explain
from database.db import (
    init_db,
    SessionLocal,
    Session,
    User as UserModel,
    Movie as MovieModel,
    Rating as RatingModel,
    Recommendation as RecommendationModel
)

# ------------------------------------------------------------------------------
# LIFESPAN & ESTRUTURA DA APLICAÇÃO
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia a inicialização dos recursos (banco de dados e modelo ML)."""
    # Inicializar banco de dados SQLite
    init_db()
    print("[API Lifespan] Banco de dados e modelo de Machine Learning prontos!")
    yield
    print("[API Lifespan] Encerrando recursos da API.")


app = FastAPI(
    title="🎬 Explainable Movie Recommender API",
    description=(
        "API RESTful para recomendação inteligente de filmes com explicações em linguagem natural "
        "baseada em fatores latentes de matrizes SVD e perfil de preferências do usuário."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Configuração de CORS para permitir requisições do frontend Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringir ao endereço do Streamlit (ex: http://localhost:8501)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency para obter a sessão do banco de dados
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------------------
# SCHEMAS PYDANTIC (DTOs)
# ------------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    database_connected: bool = Field(..., example=True)
    timestamp: str = Field(..., example="2026-08-07T11:00:00")


class RatingCreate(BaseModel):
    user_id: int = Field(..., example=1, description="ID do Usuário")
    movie_id: int = Field(..., example=50, description="ID do Filme")
    rating: float = Field(..., ge=1.0, le=5.0, example=4.5, description="Nota de 1.0 a 5.0 estrelas")


class RatingResponse(BaseModel):
    message: str = Field(..., example="Avaliação salva com sucesso.")
    user_id: int = Field(..., example=1)
    movie_id: int = Field(..., example=50)
    rating: float = Field(..., example=4.5)


class MovieSummary(BaseModel):
    movie_id: int = Field(..., example=50)
    title: str = Field(..., example="Star Wars (1977)")
    release_date: Optional[str] = Field(None, example="01-Jan-1977")
    genres: Optional[str] = Field(None, example="Action, Adventure, Sci-Fi")
    avg_rating: Optional[float] = Field(None, example=4.35)
    ratings_count: Optional[int] = Field(None, example=583)


class RecommendationItem(BaseModel):
    movie_id: int = Field(..., example=50)
    title: str = Field(..., example="Star Wars (1977)")
    predicted_rating: float = Field(..., example=4.65)
    explanation: dict = Field(..., description="Dicionário detalhado de explicabilidade")


class ExplanationResponse(BaseModel):
    user_id: int = Field(..., example=1)
    movie_id: int = Field(..., example=50)
    movie_title: str = Field(..., example="Star Wars (1977)")
    predicted_rating: float = Field(..., example=4.65)
    explanation_text: str = Field(..., example="Recomendamos 'Star Wars (1977)'...")
    details: dict = Field(..., description="Evidências completas da recomendação")


# ------------------------------------------------------------------------------
# ENDPOINTS RESTful
# ------------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Integridade"],
    summary="Verificar saúde da API e conexões"
)
def check_health(db: Session = Depends(get_db_session)):
    """Verifica se o modelo e a conexão com o SQLite estão operacionais."""
    model_exists = (MODELS_DIR / "best_model.pkl").exists()
    db_connected = True
    try:
        db.query(UserModel).first()
    except Exception:
        db_connected = False

    return {
        "status": "healthy" if (model_exists and db_connected) else "degraded",
        "model_loaded": model_exists,
        "database_connected": db_connected,
        "timestamp": datetime.now().isoformat()
    }


@app.get(
    "/recommendations/{user_id}",
    response_model=List[RecommendationItem],
    tags=["Recomendações"],
    summary="Gerar recomendações Top-K explicáveis para um usuário"
)
def get_recommendations(
    user_id: int,
    k: int = Query(10, ge=1, le=50, description="Quantidade de recomendações"),
    db: Session = Depends(get_db_session)
):
    """
    Gera uma lista com as **K melhores recomendações não vistas** para o usuário,
    acopladas com a justificativa explicativa em linguagem natural.
    Persiste as recomendações geradas no histórico do banco de dados.
    """
    try:
        recs = get_top_n_recommendations(user_id=user_id, n=k, include_explanations=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar recomendações: {str(e)}"
        )

    if not recs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhuma recomendação encontrada para o usuário #{user_id}."
        )

    # Persistir no banco de dados SQLite para histórico do sistema
    try:
        for item in recs:
            rec_entry = RecommendationModel(
                user_id=user_id,
                movie_id=item["movie_id"],
                predicted_rating=item["predicted_rating"],
                confidence=item["explanation"]["confidence"]["level"],
                explanation_text=item["explanation"]["explanation_text"]
            )
            db.add(rec_entry)
        db.commit()
    except Exception as db_err:
        db.rollback()
        print(f"[Aviso DB] Não foi possível persistir recomendação no SQLite: {db_err}")

    return recs


@app.get(
    "/explain/{user_id}/{movie_id}",
    response_model=ExplanationResponse,
    tags=["Explicabilidade"],
    summary="Obter explicação detalhada para uma recomendação específica"
)
def get_explanation(user_id: int, movie_id: int):
    """
    Retorna a evidência explicativa completa (filmes influentes do histórico,
    gêneros amados, usuários similares e texto persuasivo) para o par (usuário, filme).
    """
    try:
        exp_data = explain(user_id=user_id, movie_id=movie_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar explicação: {str(e)}"
        )

    return {
        "user_id": user_id,
        "movie_id": movie_id,
        "movie_title": exp_data["movie_title"],
        "predicted_rating": exp_data["predicted_rating"],
        "explanation_text": exp_data["explanation_text"],
        "details": exp_data
    }


@app.get(
    "/movies/popular",
    response_model=List[MovieSummary],
    tags=["Catálogo"],
    summary="Obter catálogo dos filmes mais populares e bem avaliados"
)
def get_popular_movies(
    limit: int = Query(20, ge=1, le=100, description="Número de filmes a retornar"),
    db: Session = Depends(get_db_session)
):
    """Retorna os filmes mais populares do acervo, ordenados por número de avaliações."""
    try:
        # Consulta com contagem de ratings e média
        results = (
            db.query(
                MovieModel.movie_id,
                MovieModel.movie_title,
                MovieModel.release_date,
                MovieModel.genres_str,
                func.avg(RatingModel.rating).label("avg_rating"),
                func.count(RatingModel.rating).label("ratings_count")
            )
            .join(RatingModel, MovieModel.movie_id == RatingModel.movie_id)
            .group_by(MovieModel.movie_id, MovieModel.movie_title, MovieModel.release_date, MovieModel.genres_str)
            .order_by(func.count(RatingModel.rating).desc())
            .limit(limit)
            .all()
        )

        output = []
        for r in results:
            output.append({
                "movie_id": r.movie_id,
                "title": r.movie_title,
                "release_date": r.release_date,
                "genres": r.genres_str,
                "avg_rating": round(float(r.avg_rating), 2) if r.avg_rating else None,
                "ratings_count": r.ratings_count
            })

        return output
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao consultar catálogo popular: {str(e)}"
        )


@app.post(
    "/ratings",
    response_model=RatingResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Avaliações"],
    summary="Cadastrar ou atualizar a avaliação de um filme pelo usuário"
)
def create_rating(
    rating_data: RatingCreate,
    db: Session = Depends(get_db_session)
):
    """
    Insere ou atualiza a nota de um usuário para determinado filme no banco de dados.
    """
    try:
        # Verificar se a avaliação já existe
        existing_rating = (
            db.query(RatingModel)
            .filter(RatingModel.user_id == rating_data.user_id, RatingModel.movie_id == rating_data.movie_id)
            .first()
        )

        if existing_rating:
            existing_rating.rating = rating_data.rating
            existing_rating.timestamp = int(datetime.now().timestamp())
            msg = "Avaliação atualizada com sucesso."
        else:
            new_rating = RatingModel(
                user_id=rating_data.user_id,
                movie_id=rating_data.movie_id,
                rating=rating_data.rating,
                timestamp=int(datetime.now().timestamp())
            )
            db.add(new_rating)
            msg = "Avaliação criada com sucesso."

        db.commit()

        return {
            "message": msg,
            "user_id": rating_data.user_id,
            "movie_id": rating_data.movie_id,
            "rating": rating_data.rating
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao salvar avaliação: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
