import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(__file__))
from api.main import app

client = TestClient(app)

def test_get_tickers():
    response = client.get("/tickers")
    assert response.status_code == 200
    data = response.json()
    assert "tickers" in data
    assert isinstance(data["tickers"], list)
