import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# 1. Testing Home Route
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

#2. Testing Valid Prediction
def test_predict_valid_image():
    with open("sample.JPG", "rb") as image_file:
        response = client.post("/predict", files={"file": ("sample.JPG", image_file, "image/jpeg")})
    assert response.status_code == 200
    assert "Disease" in response.json()
    assert "Recommendation" in response.json()

# 3. Testing Invalid File Type
def test_predict_invalid_filetype():
    with open("sample.txt", "w") as f:
        f.write("This is not an image")
    with open("sample.txt", "rb") as f:
        response = client.post("/predict", files={"file": ("sample.txt", f, "text/plain")})
    assert response.status_code == 400
    assert "error" in response.json()

#  4. Testing Low Confidence Image 
def test_predict_low_confidence(monkeypatch):
    def fake_predict(_):
        return [[0.01] * 38]  # Simulating a low confidence prediction
    monkeypatch.setattr("main.model.predict", fake_predict)

    with open("sample.JPG", "rb") as image_file:
        response = client.post("/predict", files={"file": ("sample.JPG", image_file, "image/jpeg")})
    assert response.status_code == 400
    assert "Low confidence" in response.json()["error"]

#  5. Testing Missing File
def test_predict_no_file():
    response = client.post("/predict", files={})
    assert response.status_code == 422  
