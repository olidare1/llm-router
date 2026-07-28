from fastapi.testclient import TestClient
from main import app, route_prompt, ModelType

client = TestClient(app)

def test_routing_logic():
    # Test short prompt (< 50 chars) -> FAST
    assert route_prompt("Hallo wie gehts?") == ModelType.FAST
    
    # Test prompt with "zusammenfassen" keyword -> FAST even if long
    long_summarize = "Bitte kannst du mir diesen langen Text zusammenfassen: " + "a" * 100
    assert route_prompt(long_summarize) == ModelType.FAST

    # Test long complex prompt -> HEAVY
    long_complex = "Analysiere diesen Python-Code und erstelle eine detaillierte Architekturübersicht mit allen Klassen und Methoden."
    assert route_prompt(long_complex) == ModelType.HEAVY

def test_api_endpoint_fast():
    response = client.post("/v1/chat/completions", json={"prompt": "Kurze Frage", "user_id": "usr_123"})
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == ModelType.FAST.value
    assert "response_text" in data
    assert "latency_ms" in data
    assert data["latency_ms"] >= 300  # at least ~300ms sleep

def test_api_endpoint_heavy():
    response = client.post("/v1/chat/completions", json={
        "prompt": "Das ist ein langer und sehr komplexer Prompt, der nicht das Zauberwort enthält und über 50 Zeichen lang ist."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == ModelType.HEAVY.value
    assert "response_text" in data
    assert "latency_ms" in data
    assert data["latency_ms"] >= 2000  # at least ~2000ms sleep
