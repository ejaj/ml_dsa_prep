from calculator import add
import pytest

def test_add():
    assert add(2,3) == 5
    assert add(-1, 1) == 0
    assert add(1,1) != 3

@pytest.fixture
def db_connect():
    print("SETUP: Connecting to DB")
    conn = "Connection db"
    yield conn
    print("TEARDOWN: Closing DB")
    conn.close()

@pytest.fixture
def auth_token():
    return "secret-token"
# pytest --junitxml=results.xml
from unittest.mock import patch

def test_app_call():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"status":"ok"}
        result = my_api_call()
        assert result["status"] == "ok"

@pytest.mark.integration
def test_api_database_flow(client, db_session):
    response = client.get('users/1')
    assert response.status_code == 200
    assert db_session.query(User).get(1) is not None