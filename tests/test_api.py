from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


# ============================================================
# 1. TEST ROOT
# ============================================================

def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Breast Cancer SVM API"
    assert data["version"] == "1.0.0"


# ============================================================
# 2. TEST HEALTH
# ============================================================

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"


# ============================================================
# 3. TEST METADATA
# ============================================================

def test_metadata():
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert "feature_names" in data
    assert len(data["feature_names"]) == 30

    assert data["class_mapping"]["0"] == "malignant"
    assert data["class_mapping"]["1"] == "benign"


# ============================================================
# 4. DỮ LIỆU MẪU
# ============================================================

def sample_features():
    return {
        "mean radius": 17.99,
        "mean texture": 10.38,
        "mean perimeter": 122.8,
        "mean area": 1001.0,
        "mean smoothness": 0.1184,
        "mean compactness": 0.2776,
        "mean concavity": 0.3001,
        "mean concave points": 0.1471,
        "mean symmetry": 0.2419,
        "mean fractal dimension": 0.07871,
        "radius error": 1.095,
        "texture error": 0.9053,
        "perimeter error": 8.589,
        "area error": 153.4,
        "smoothness error": 0.006399,
        "compactness error": 0.04904,
        "concavity error": 0.05373,
        "concave points error": 0.01587,
        "symmetry error": 0.03003,
        "fractal dimension error": 0.006193,
        "worst radius": 25.38,
        "worst texture": 17.33,
        "worst perimeter": 184.6,
        "worst area": 2019.0,
        "worst smoothness": 0.1622,
        "worst compactness": 0.6656,
        "worst concavity": 0.7119,
        "worst concave points": 0.2654,
        "worst symmetry": 0.4601,
        "worst fractal dimension": 0.1189
    }


# ============================================================
# 5. TEST PREDICT
# ============================================================

def test_predict():
    response = client.post(
        "/predict",
        json={
            "features": sample_features()
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] in [0, 1]

    assert data["predicted_label"] in [
        "malignant",
        "benign"
    ]

    assert 0 <= data["probability_malignant"] <= 1
    assert 0 <= data["probability_benign"] <= 1

    assert (
        abs(
            data["probability_malignant"]
            + data["probability_benign"]
            - 1
        ) < 1e-6
    )

    assert data["model_version"] == "1.0.0"


# ============================================================
# 6. TEST THIẾU FEATURE
# ============================================================

def test_missing_feature():

    features = sample_features()

    features.pop("mean radius")

    response = client.post(
        "/predict",
        json={
            "features": features
        }
    )

    assert response.status_code == 422


# ============================================================
# 7. TEST FEATURE THỪA
# ============================================================

def test_extra_feature():

    features = sample_features()

    features["extra feature"] = 123.0

    response = client.post(
        "/predict",
        json={
            "features": features
        }
    )

    assert response.status_code == 422


# ============================================================
# 8. TEST GIÁ TRỊ KHÔNG HỢP LỆ
# ============================================================

def test_invalid_value():

    features = sample_features()

    features["mean radius"] = "abc"

    response = client.post(
        "/predict",
        json={
            "features": features
        }
    )

    assert response.status_code == 422