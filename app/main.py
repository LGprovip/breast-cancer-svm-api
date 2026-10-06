from pathlib import Path
import json
import math

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "breast_cancer_svm.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

TEMPLATE_PATH = BASE_DIR / "app" / "templates" / "index.html"
STATIC_DIR = BASE_DIR / "app" / "static"


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Breast Cancer SVM API",
    description="SVM model for Breast Cancer Wisconsin dataset",
    version="1.0.0",
)


# Static files
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


# =========================================================
# FEATURE NAMES
# =========================================================

FEATURE_NAMES = metadata["feature_names"]

CLASS_MAPPING = metadata["class_mapping"]


# =========================================================
# PYDANTIC MODELS
# =========================================================

class PredictionRequest(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    warning: str


# =========================================================
# HELPERS
# =========================================================

def build_vector(features: dict[str, float]):

    missing = [
        feature
        for feature in FEATURE_NAMES
        if feature not in features
    ]

    extra = [
        feature
        for feature in features
        if feature not in FEATURE_NAMES
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing features",
                "features": missing,
            },
        )

    if extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unknown features",
                "features": extra,
            },
        )

    values = []

    for feature in FEATURE_NAMES:

        value = features[feature]

        if not isinstance(value, (int, float)):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"Invalid value for feature: {feature}"
                },
            )

        if not math.isfinite(float(value)):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"Value must be finite: {feature}"
                },
            )

        values.append(float(value))

    return np.array(values).reshape(1, -1)


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    return HTMLResponse(content=html)


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": metadata.get(
            "model_version",
            "1.0.0"
        ),
    }


# =========================================================
# METADATA
# =========================================================

@app.get("/metadata")
def get_metadata():

    return metadata


# =========================================================
# PREDICT
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    X = build_vector(request.features)

    prediction = int(model.predict(X)[0])

    probabilities = model.predict_proba(X)[0]

    classes = model.named_steps["svc"].classes_

    probability_malignant = 0.0
    probability_benign = 0.0

    for class_value, probability in zip(
        classes,
        probabilities
    ):

        if int(class_value) == 0:
            probability_malignant = float(probability)

        elif int(class_value) == 1:
            probability_benign = float(probability)

    predicted_label = CLASS_MAPPING[str(prediction)]

    return {
        "predicted_class": prediction,
        "predicted_label": predicted_label,
        "probability_malignant": probability_malignant,
        "probability_benign": probability_benign,
        "model_version": metadata.get(
            "model_version",
            "1.0.0"
        ),
        "warning": metadata.get(
            "warning",
            "Educational use only; not a medical diagnosis."
        ),
    }