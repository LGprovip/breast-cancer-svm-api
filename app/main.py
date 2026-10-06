from pathlib import Path
import json

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "artifacts" / "breast_cancer_svm.joblib"
META_PATH = BASE_DIR / "artifacts" / "metadata.json"


# ============================================================
# 2. LOAD MODEL VÀ METADATA
# ============================================================

model = joblib.load(MODEL_PATH)

metadata = json.loads(
    META_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# 3. KHỞI TẠO FASTAPI
# ============================================================

app = FastAPI(
    title="Breast Cancer SVM API",
    version=metadata["model_version"],
    description="Educational demonstration only"
)


# ============================================================
# 4. PYDANTIC MODEL
# ============================================================

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Exactly 30 named numeric features"
    )


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    warning: str


# ============================================================
# 5. BUILD VECTOR
# ============================================================

def build_vector(
    payload: PredictionRequest
) -> np.ndarray:

    expected_features = metadata["feature_names"]

    received_features = set(
        payload.features.keys()
    )

    expected_features_set = set(
        expected_features
    )

    missing_features = (
        expected_features_set
        - received_features
    )

    extra_features = (
        received_features
        - expected_features_set
    )

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing features",
                "features": sorted(
                    missing_features
                )
            }
        )

    if extra_features:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Extra features",
                "features": sorted(
                    extra_features
                )
            }
        )

    values = [
        payload.features[name]
        for name in expected_features
    ]

    vector = np.asarray(
        values,
        dtype=float
    ).reshape(1, -1)

    if not np.isfinite(vector).all():
        raise HTTPException(
            status_code=422,
            detail="Features must be finite numbers."
        )

    return vector


# ============================================================
# 6. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Breast Cancer SVM API",
        "version": metadata["model_version"],
        "docs": "/docs"
    }


# ============================================================
# 7. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": metadata["model_version"]
    }


# ============================================================
# 8. METADATA ENDPOINT
# ============================================================

@app.get("/metadata")
def get_metadata():

    return metadata


# ============================================================
# 9. PREDICT ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    payload: PredictionRequest
):

    x = build_vector(payload)

    predicted_class = int(
        model.predict(x)[0]
    )

    probabilities = model.predict_proba(x)[0]

    classes = list(
        model.named_steps["svc"].classes_
    )

    probability_map = {
        int(c): float(v)
        for c, v in zip(
            classes,
            probabilities
        )
    }

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=metadata[
            "class_mapping"
        ][str(predicted_class)],
        probability_malignant=probability_map[0],
        probability_benign=probability_map[1],
        model_version=metadata[
            "model_version"
        ],
        warning=metadata[
            "warning"
        ]
    )