from pathlib import Path
import json
import math

import joblib
import numpy as np

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Form,
)

from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
)

from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from starlette.middleware.sessions import SessionMiddleware

from app.database import (
    init_database,
    create_user,
    get_user_by_username,
)

from app.auth import (
    hash_password,
    verify_password,
)


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = (
    ARTIFACT_DIR /
    "breast_cancer_svm.joblib"
)

METADATA_PATH = (
    ARTIFACT_DIR /
    "metadata.json"
)

TEMPLATE_DIR = (
    BASE_DIR /
    "app" /
    "templates"
)

TEMPLATE_PATH = (
    TEMPLATE_DIR /
    "index.html"
)

LOGIN_TEMPLATE_PATH = (
    TEMPLATE_DIR /
    "login.html"
)

REGISTER_TEMPLATE_PATH = (
    TEMPLATE_DIR /
    "register.html"
)

STATIC_DIR = (
    BASE_DIR /
    "app" /
    "static"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Breast Cancer SVM API",
    description=(
        "SVM model for "
        "Breast Cancer Wisconsin dataset"
    ),
    version="1.0.0",
)


# =========================================================
# SESSION
# =========================================================

# IMPORTANT:
# Khi deploy thật, nên thay secret_key này
# bằng một chuỗi bí mật dài và ngẫu nhiên.

app.add_middleware(
    SessionMiddleware,
    secret_key=(
        "breast-cancer-svm-demo-"
        "change-this-secret-key-"
        "2026"
    ),
    max_age=60 * 60 * 24 * 7,
    same_site="lax",
    https_only=False,
)


# =========================================================
# DATABASE
# =========================================================

init_database()


# =========================================================
# STATIC FILES
# =========================================================

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
# AUTH HELPERS
# =========================================================

def is_logged_in(request: Request) -> bool:

    return (
        request.session.get("user_id")
        is not None
    )


def get_current_username(
    request: Request
):

    return request.session.get(
        "username"
    )


# =========================================================
# TEMPLATE HELPERS
# =========================================================

def read_template(
    template_path: Path
) -> str:

    if not template_path.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                f"Template not found: "
                f"{template_path}"
            ),
        )

    with open(
        template_path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# =========================================================
# FEATURE VECTOR
# =========================================================

def build_vector(
    features: dict[str, float]
):

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

        if not isinstance(
            value,
            (int, float)
        ):

            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Invalid value for "
                        f"feature: {feature}"
                    )
                },
            )

        if not math.isfinite(
            float(value)
        ):

            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Value must be finite: "
                        f"{feature}"
                    )
                },
            )

        values.append(
            float(value)
        )

    return np.array(
        values
    ).reshape(1, -1)


# =========================================================
# HOME PAGE
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home(
    request: Request
):

    # Nếu chưa đăng nhập
    # chuyển về trang Login

    if not is_logged_in(request):

        return RedirectResponse(
            url="/login",
            status_code=303,
        )

    html = read_template(
        TEMPLATE_PATH
    )

    # Cho phép index.html sử dụng
    # {{ username }}

    username = get_current_username(
        request
    )

    html = html.replace(
        "{{ username }}",
        str(username or "User")
    )

    return HTMLResponse(
        content=html
    )


# =========================================================
# REGISTER - GET
# =========================================================

@app.get(
    "/register",
    response_class=HTMLResponse
)
def register_page():

    html = read_template(
        REGISTER_TEMPLATE_PATH
    )

    html = html.replace(
        "{{ error }}",
        ""
    )

    return HTMLResponse(
        content=html
    )

# =========================================================
# REGISTER - POST
# =========================================================

@app.post(
    "/register",
    response_class=HTMLResponse
)
def register(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):

    # Remove spaces

    username = username.strip()

    email = email.strip().lower()


    # Username validation

    if len(username) < 3:

        html = read_template(
            REGISTER_TEMPLATE_PATH
        )

        return HTMLResponse(
            content=html.replace(
                "{{ error }}",
                "Username must contain at least 3 characters."
            ),
            status_code=400,
        )


    # Email validation

    if (
        "@" not in email
        or "." not in email
    ):

        html = read_template(
            REGISTER_TEMPLATE_PATH
        )

        return HTMLResponse(
            content=html.replace(
                "{{ error }}",
                "Please enter a valid email address."
            ),
            status_code=400,
        )


    # Password validation

    if len(password) < 6:

        html = read_template(
            REGISTER_TEMPLATE_PATH
        )

        return HTMLResponse(
            content=html.replace(
                "{{ error }}",
                "Password must contain at least 6 characters."
            ),
            status_code=400,
        )


    # Confirm password

    if password != confirm_password:

        html = read_template(
            REGISTER_TEMPLATE_PATH
        )

        return HTMLResponse(
            content=html.replace(
                "{{ error }}",
                "Passwords do not match."
            ),
            status_code=400,
        )


    # Check existing username

    existing_user = (
        get_user_by_username(
            username
        )
    )

    if existing_user:

        html = read_template(
            REGISTER_TEMPLATE_PATH
        )

        return HTMLResponse(
            content=html.replace(
                "{{ error }}",
                "Username already exists."
            ),
            status_code=400,
        )


    # Hash password

    password_hash = hash_password(
        password
    )


    # Create user

    user_id = create_user(
        username=username,
        email=email,
        password_hash=password_hash,
    )


    if user_id is None:

        html = read_template(
            REGISTER_TEMPLATE_PATH
        )

        return HTMLResponse(
            content=html.replace(
                "{{ error }}",
                "Username or email already exists."
            ),
            status_code=400,
        )


    # Register successful
    # Redirect to login

    return RedirectResponse(
        url="/login?registered=1",
        status_code=303,
    )


# =========================================================
# LOGIN - GET
# =========================================================

@app.get(
    "/login",
    response_class=HTMLResponse
)
def login_page(
    request: Request
):

    # Nếu đã đăng nhập
    # không cần Login lại

    if is_logged_in(request):

        return RedirectResponse(
            url="/",
            status_code=303,
        )

    html = read_template(
        LOGIN_TEMPLATE_PATH
    )

    registered = request.query_params.get(
        "registered"
    )

    if registered == "1":

        html = html.replace(
            "{{ error }}",
            "Account created successfully. Please login."
        )

    else:

        html = html.replace(
            "{{ error }}",
            ""
        )

    return HTMLResponse(
        content=html
    )


# =========================================================
# LOGIN - POST
# =========================================================

@app.post(
    "/login",
    response_class=HTMLResponse
)
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):

    username = username.strip()


    # Find user

    user = get_user_by_username(
        username
    )


    # User does not exist

    if not user:

        html = read_template(
            LOGIN_TEMPLATE_PATH
        )

        return HTMLResponse(
            content=html.replace(
                "{{ error }}",
                "Invalid username or password."
            ),
            status_code=401,
        )


    # Verify password

    password_valid = verify_password(
        password,
        user["password_hash"]
    )


    if not password_valid:

        html = read_template(
            LOGIN_TEMPLATE_PATH
        )

        return HTMLResponse(
            content=html.replace(
                "{{ error }}",
                "Invalid username or password."
            ),
            status_code=401,
        )


    # =====================================================
    # CREATE SESSION
    # =====================================================

    request.session.clear()

    request.session["user_id"] = (
        user["id"]
    )

    request.session["username"] = (
        user["username"]
    )

    request.session["email"] = (
        user["email"]
    )


    # Go to prediction page

    return RedirectResponse(
        url="/",
        status_code=303,
    )


# =========================================================
# LOGOUT
# =========================================================

@app.get(
    "/logout"
)
def logout(
    request: Request
):

    request.session.clear()

    return RedirectResponse(
        url="/login",
        status_code=303,
    )


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
def predict(
    request: Request,
    prediction_request: PredictionRequest
):

    # =====================================================
    # REQUIRE LOGIN
    # =====================================================

    if not is_logged_in(request):

        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )


    # =====================================================
    # BUILD VECTOR
    # =====================================================

    X = build_vector(
        prediction_request.features
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = int(
        model.predict(X)[0]
    )


    probabilities = (
        model.predict_proba(X)[0]
    )


    classes = (
        model.named_steps["svc"].classes_
    )


    probability_malignant = 0.0

    probability_benign = 0.0


    for (
        class_value,
        probability
    ) in zip(
        classes,
        probabilities
    ):

        if int(class_value) == 0:

            probability_malignant = float(
                probability
            )

        elif int(class_value) == 1:

            probability_benign = float(
                probability
            )


    predicted_label = (
        CLASS_MAPPING[
            str(prediction)
        ]
    )


    return {

        "predicted_class":
            prediction,

        "predicted_label":
            predicted_label,

        "probability_malignant":
            probability_malignant,

        "probability_benign":
            probability_benign,

        "model_version":
            metadata.get(
                "model_version",
                "1.0.0"
            ),

        "warning":
            metadata.get(
                "warning",
                "Educational use only; not a medical diagnosis."
            ),
    }
