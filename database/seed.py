"""
Script de Povoamento (Seed) do Banco de Dados e Teste de Operações CRUD.

1. Carrega os DataFrames de data/processed/ (users, movies, ratings).
2. Popula as tabelas do SQLite em lote (bulk insert).
3. Executa testes unitários de operações CRUD (Create, Read, Update, Delete) em todas as tabelas.
"""

import sys
from pathlib import Path
import pandas as pd

# Adiciona o diretório raiz ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DATA_DIR, GENRES
from database.db import (
    init_db,
    SessionLocal,
    User,
    Movie,
    Rating,
    Recommendation
)


def seed_database() -> None:
    """Popula o banco de dados SQLite a partir dos arquivos CSV processados em data/processed/."""
    print("=== INICIANDO SEED DO BANCO DE DADOS ===")
    init_db()
    session = SessionLocal()

    try:
        # 1. Povoar Usuários (se a tabela estiver vazia)
        if session.query(User).count() == 0:
            users_df = pd.read_csv(PROCESSED_DATA_DIR / "processed_users.csv")
            users_mappings = users_df.to_dict(orient="records")
            session.bulk_insert_mappings(User, users_mappings)
            session.commit()
            print(f"[Seed] Inseridos {len(users_mappings):,} usuários em 'users'.")
        else:
            print("[Seed] Tabela 'users' já possui registros. Povoamento ignorado.")

        # 2. Povoar Filmes (se a tabela estiver vazia)
        if session.query(Movie).count() == 0:
            movies_df = pd.read_csv(PROCESSED_DATA_DIR / "processed_movies.csv")

            # Construir genres_str para cada filme (ex: "Action, Sci-Fi, War")
            movies_records = []
            for _, row in movies_df.iterrows():
                active_genres = [g for g in GENRES if g in row and row[g] == 1]
                genres_str = ", ".join(active_genres) if active_genres else "Unknown"

                movies_records.append({
                    "movie_id": int(row["movie_id"]),
                    "movie_title": str(row["movie_title"]),
                    "release_date": str(row["release_date"]),
                    "genres_str": genres_str
                })

            session.bulk_insert_mappings(Movie, movies_records)
            session.commit()
            print(f"[Seed] Inseridos {len(movies_records):,} filmes em 'movies'.")
        else:
            print("[Seed] Tabela 'movies' já possui registros. Povoamento ignorado.")

        # 3. Povoar Avaliações (se a tabela estiver vazia)
        if session.query(Rating).count() == 0:
            ratings_df = pd.read_csv(PROCESSED_DATA_DIR / "processed_ratings.csv")
            ratings_mappings = ratings_df[["user_id", "item_id", "rating", "timestamp"]].rename(
                columns={"item_id": "movie_id"}
            ).to_dict(orient="records")

            session.bulk_insert_mappings(Rating, ratings_mappings)
            session.commit()
            print(f"[Seed] Inseridas {len(ratings_mappings):,} avaliações em 'ratings'.")
        else:
            print("[Seed] Tabela 'ratings' já possui registros. Povoamento ignorado.")

    except Exception as e:
        session.rollback()
        print(f"[Erro no Seed]: {e}")
        raise e
    finally:
        session.close()

    print("=== SEED CONCLUÍDO COM SUCESSO! ===")


def test_crud_operations() -> None:
    """Executa testes de operações CRUD (Create, Read, Update, Delete) nas tabelas do ORM."""
    print("\n=== EXECUTANDO TESTES DE OPERAÇÕES CRUD ===")
    session = SessionLocal()

    try:
        # --- 1. CREATE ---
        test_rec = Recommendation(
            user_id=1,
            movie_id=50,
            predicted_rating=4.85,
            confidence="Alta",
            explanation_text="Teste de recomendação explicável via CRUD."
        )
        session.add(test_rec)
        session.commit()
        session.refresh(test_rec)
        rec_id = test_rec.id
        print(f"[CRUD - CREATE] Inserida recomendação id={rec_id} com sucesso.")

        # --- 2. READ ---
        read_rec = session.query(Recommendation).filter(Recommendation.id == rec_id).first()
        assert read_rec is not None, "Falha na leitura da recomendação."
        assert read_rec.predicted_rating == 4.85, "Valor da nota prevista não bate."
        print(f"[CRUD - READ] Consulta recomendação id={rec_id}: User #{read_rec.user_id}, Movie #{read_rec.movie_id}, Rating: {read_rec.predicted_rating}.")

        # --- 3. UPDATE ---
        read_rec.predicted_rating = 4.99
        read_rec.confidence = "Muito Alta"
        session.commit()

        updated_rec = session.query(Recommendation).filter(Recommendation.id == rec_id).first()
        assert updated_rec.predicted_rating == 4.99, "Falha ao atualizar nota prevista."
        print(f"[CRUD - UPDATE] Atualizada recomendação id={rec_id} para nota {updated_rec.predicted_rating}.")

        # --- 4. DELETE ---
        session.delete(updated_rec)
        session.commit()

        deleted_rec = session.query(Recommendation).filter(Recommendation.id == rec_id).first()
        assert deleted_rec is None, "Falha ao deletar recomendação."
        print(f"[CRUD - DELETE] Removida recomendação id={rec_id} com sucesso.")

        print("\n=== TODAS AS OPERAÇÕES CRUD FORAM VALIDADAS COM SUCESSO! ===")

    except Exception as e:
        session.rollback()
        print(f"[Erro no teste CRUD]: {e}")
        raise e
    finally:
        session.close()


if __name__ == "__main__":
    seed_database()
    test_crud_operations()
