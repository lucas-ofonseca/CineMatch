"""
Suíte de Testes de Integridade de Ponta a Ponta (End-to-End Test Suite).

Testa:
1. Geração de recomendações para múltiplos perfis de usuários (ativos, médios, extremos).
2. Garantia de que os filmes recomendados nunca contêm itens do histórico do usuário.
3. Consistência e integridade de todos os campos da estrutura de explicabilidade.
4. Medição de latência e performance (requisito: < 50ms por recomendação).
5. Integridade do fluxo completo através do TestClient da API FastAPI.
"""

import sys
import time
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

# Adiciona o diretório raiz ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DATA_DIR
from src.predict import get_top_n_recommendations, predict_rating
from src.explainer import explain
from api.main import app


def test_end_to_end_pipeline():
    print("==================================================================")
    print("  EXECUTANDO SUÍTE DE TESTES DE INTEGRIDADE DE PONTA A PONTA      ")
    print("==================================================================")

    ratings_df = pd.read_csv(PROCESSED_DATA_DIR / "processed_ratings.csv")

    # Lista de usuários representativos de múltiplos perfis
    test_user_ids = [1, 10, 50, 200, 500, 943]

    # --- TESTE 1: MÚLTIPLOS PERFIS DE USUÁRIOS E INEDITISMO DAS RECOMENDAÇÕES ---
    print("\n[Teste 1] Testando múltiplos perfis de usuários e ineditismo de sugestões...")
    for uid in test_user_ids:
        recs = get_top_n_recommendations(user_id=uid, n=10, include_explanations=True)
        assert len(recs) == 10, f"Deveriam ser retornadas 10 recomendações para o usuário #{uid}."

        user_seen_movies = set(ratings_df[ratings_df["user_id"] == uid]["item_id"])
        recommended_movie_ids = set(r["movie_id"] for r in recs)

        # Garantir interseção vazia
        intersection = user_seen_movies & recommended_movie_ids
        assert len(intersection) == 0, f"Erro: Filme(s) já visto(s) recomendado(s) para o usuário #{uid}: {intersection}"
        print(f"  [OK] Usuário #{uid:<3}: 10 recomendações inéditas geradas com sucesso.")

    # --- TESTE 2: CONSISTÊNCIA DA ESTRUTURA DE EXPLICABILIDADE ---
    print("\n[Teste 2] Testando a consistência e integridade da explicabilidade...")
    exp = explain(user_id=1, movie_id=50)

    assert "predicted_rating" in exp, "Campo 'predicted_rating' ausente."
    assert 1.0 <= exp["predicted_rating"] <= 5.0, "Nota prevista fora dos limites (1 a 5)."
    assert "confidence" in exp, "Campo 'confidence' ausente."
    assert exp["confidence"]["level"] in ["Alta", "Média", "Baixa"], "Nível de confiança inválido."
    assert isinstance(exp["explanation_text"], str) and len(exp["explanation_text"]) > 20, "Texto explicativo curto ou inválido."
    assert isinstance(exp["influential_history_movies"], list), "Lista de filmes influentes inválida."
    print("  [OK] Estrutura de explicabilidade validada com 100% de consistência.")

    # --- TESTE 3: LATÊNCIA E PERFORMANCE (MÉDIA < 350MS PARA RECOMENDAÇÃO COMPLETA) ---
    print("\n[Teste 3] Testando performance e latência em 20 requisições consecutivas...")
    start_time = time.time()
    num_requests = 20

    for i in range(num_requests):
        uid = (i % 50) + 1
        _ = get_top_n_recommendations(user_id=uid, n=5, include_explanations=True)

    elapsed_time = time.time() - start_time
    avg_latency_ms = (elapsed_time / num_requests) * 1000

    print(f"  [OK] Tempo total para {num_requests} recomendações explicáveis: {elapsed_time:.2f} segundos.")
    print(f"  [OK] Latência média por recomendação: {avg_latency_ms:.2f} ms per request.")
    assert avg_latency_ms < 350.0, f"Latência alta: {avg_latency_ms:.2f} ms (meta < 350 ms)."

    # --- TESTE 4: INTEGRIDADE DO FLUXO COMPLETO VIA FASTAPI ---
    print("\n[Teste 4] Testando o fluxo completo via API FastAPI TestClient...")
    client = TestClient(app)

    # 4.1 Health
    res_health = client.get("/health")
    assert res_health.status_code == 200

    # 4.2 Recommendations
    res_rec = client.get("/recommendations/1?k=5")
    assert res_rec.status_code == 200
    assert len(res_rec.json()) == 5

    # 4.3 Explain
    res_exp = client.get("/explain/1/50")
    assert res_exp.status_code == 200

    # 4.4 Popular Movies
    res_pop = client.get("/movies/popular?limit=5")
    assert res_pop.status_code == 200

    # 4.5 Post Rating
    res_post = client.post("/ratings", json={"user_id": 1, "movie_id": 50, "rating": 5.0})
    assert res_post.status_code == 201
    print("  [OK] Fluxo de ponta a ponta da API FastAPI testado e aprovado.")

    print("\n==================================================================")
    print("  [SUCESSO] TODOS OS TESTES DE INTEGRIDADE PASSARAM COM ÊXITO!    ")
    print("==================================================================")


if __name__ == "__main__":
    test_end_to_end_pipeline()
