"""
Script de Teste de Integridade Automatizado para os Endpoints da API FastAPI.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app


def run_api_tests():
    print("=== INICIANDO TESTES AUTOMATIZADOS DA API FASTAPI ===")
    client = TestClient(app)

    # 1. Teste GET /health
    resp = client.get("/health")
    print(f"\n1. GET /health -> Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

    # 2. Teste GET /recommendations/1?k=3
    resp = client.get("/recommendations/1?k=3")
    print(f"\n2. GET /recommendations/1?k=3 -> Status: {resp.status_code}")
    recs = resp.json()
    print(f"   Retornadas {len(recs)} recomendações. Exemplo #1: {recs[0]['title']} (Nota: {recs[0]['predicted_rating']})")
    assert resp.status_code == 200
    assert len(recs) == 3
    assert "explanation" in recs[0]

    # 3. Teste GET /explain/1/50
    resp = client.get("/explain/1/50")
    print(f"\n3. GET /explain/1/50 -> Status: {resp.status_code}")
    exp = resp.json()
    print(f"   Filme: {exp['movie_title']} | Nota Estimada: {exp['predicted_rating']}")
    print(f"   Texto Explicativo: {exp['explanation_text']}")
    assert resp.status_code == 200
    assert exp["user_id"] == 1
    assert exp["movie_id"] == 50

    # 4. Teste GET /movies/popular?limit=5
    resp = client.get("/movies/popular?limit=5")
    print(f"\n4. GET /movies/popular?limit=5 -> Status: {resp.status_code}")
    pop = resp.json()
    print(f"   Retornados {len(pop)} filmes populares. Top 1: {pop[0]['title']} ({pop[0]['ratings_count']} avaliações)")
    assert resp.status_code == 200
    assert len(pop) == 5

    # 5. Teste POST /ratings
    payload = {"user_id": 1, "movie_id": 50, "rating": 5.0}
    resp = client.post("/ratings", json=payload)
    print(f"\n5. POST /ratings -> Status: {resp.status_code}")
    print(f"   Response: {resp.json()}")
    assert resp.status_code == 201
    assert resp.json()["user_id"] == 1
    assert resp.json()["rating"] == 5.0

    print("==================================================================")
    print("[SUCESSO] TODOS OS 5 ENDPOINTS DA API FASTAPI PASSARAM NOS TESTES 100%!")
    print("==================================================================")


if __name__ == "__main__":
    run_api_tests()
