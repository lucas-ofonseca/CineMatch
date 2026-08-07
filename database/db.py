"""
Módulo de Banco de Dados com SQLAlchemy ORM (SQLite).

Define o schema relacional para:
1. User (Tabela 'users')
2. Movie (Tabela 'movies')
3. Rating (Tabela 'ratings')
4. Recommendation (Tabela 'recommendations')

Princípios: SOLID (Single Responsibility), DRY, Clean Code.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Generator

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# Adiciona o diretório raiz ao sys.path para importar src.config
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATABASE_URL

# Engine do SQLAlchemy configurado para o SQLite local
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False
)

# Fábrica de Sessões
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Classe Base Declarativa
Base = declarative_base()


class User(Base):
    """Modelo ORM representando a tabela 'users'."""
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True)
    occupation = Column(String(50), nullable=True)
    zip_code = Column(String(20), nullable=True)

    # Relacionamentos
    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="user", cascade="all, delete-orphan")


class Movie(Base):
    """Modelo ORM representando a tabela 'movies'."""
    __tablename__ = "movies"

    movie_id = Column(Integer, primary_key=True, index=True)
    movie_title = Column(String(255), nullable=False)
    release_date = Column(String(50), nullable=True)
    genres_str = Column(String(255), nullable=True)

    # Relacionamentos
    ratings = relationship("Rating", back_populates="movie", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="movie", cascade="all, delete-orphan")


class Rating(Base):
    """Modelo ORM representando a tabela 'ratings'."""
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.movie_id"), nullable=False, index=True)
    rating = Column(Float, nullable=False)
    timestamp = Column(Integer, nullable=True)

    # Relacionamentos
    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")


class Recommendation(Base):
    """Modelo ORM representando a tabela 'recommendations' (histórico de sugestões explicáveis)."""
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.movie_id"), nullable=False, index=True)
    predicted_rating = Column(Float, nullable=False)
    confidence = Column(String(20), nullable=True)
    explanation_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relacionamentos
    user = relationship("User", back_populates="recommendations")
    movie = relationship("Movie", back_populates="recommendations")


def init_db() -> None:
    """Cria todas as tabelas no banco de dados SQLite se ainda não existirem."""
    Base.metadata.create_all(bind=engine)
    print(f"[Database] Schema SQLite inicializado com sucesso em: {DATABASE_URL}")


def get_db() -> Generator[Session, None, None]:
    """Gerador de sessão do banco de dados (Dependency Injection pattern)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
