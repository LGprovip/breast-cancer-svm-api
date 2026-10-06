from pathlib import Path
import json

import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

bundle = load_breast_cancer(as_frame=True)

X = bundle.data
y = bundle.target

print("Kích thước dữ liệu:", X.shape)
print("\nPhân bố nhãn:")
print(y.value_counts())

print("\nTên các lớp:")
print(bundle.target_names)

print("\nSố lượng giá trị thiếu:")
print(X.isna().sum().sum())

print("\nThống kê dữ liệu:")
print(X.describe().T)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nKích thước tập train:", X_train.shape)
print("Kích thước tập test:", X_test.shape)


# ============================================================
# 3. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 4. BUILD SVM PIPELINE
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])


# ============================================================
# 5. PARAMETER GRID
# ============================================================

param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1]
}


# ============================================================
# 6. GRID SEARCH
# ============================================================

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="recall_macro",
    cv=cv,
    n_jobs=-1,
    refit=True
)

print("\nĐang huấn luyện mô hình...")

search.fit(X_train, y_train)

model = search.best_estimator_

print("\nHuấn luyện hoàn tất!")
print("Best parameters:", search.best_params_)
print("Best CV score:", search.best_score_)


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

pred = model.predict(X_test)

proba_malignant = model.predict_proba(X_test)[:, 0]

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        pred,
        labels=[0, 1]
    )
)


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        pred,
        labels=[0, 1],
        target_names=["malignant", "benign"]
    )
)


print("\nSensitivity:", recall_score(
    y_test,
    pred,
    pos_label=0
))


print(
    "Precision malignant:",
    precision_score(
        y_test,
        pred,
        pos_label=0
    )
)


print(
    "ROC-AUC malignant:",
    roc_auc_score(
        (y_test == 0).astype(int),
        proba_malignant
    )
)


# ============================================================
# 8. SAVE MODEL
# ============================================================

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"

ARTIFACT_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH = ARTIFACT_DIR / "breast_cancer_svm.joblib"

joblib.dump(
    model,
    MODEL_PATH
)

print("\nĐã lưu model tại:")
print(MODEL_PATH)


# ============================================================
# 9. SAVE METADATA
# ============================================================

metadata = {
    "model_name": "breast-cancer-svm-rbf",
    "model_version": "1.0.0",
    "feature_names": list(X.columns),
    "class_mapping": {
        "0": "malignant",
        "1": "benign"
    },
    "best_params": search.best_params_,
    "sklearn_version": sklearn.__version__,
    "warning": "Educational use only; not a medical diagnosis."
}


META_PATH = ARTIFACT_DIR / "metadata.json"

META_PATH.write_text(
    json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print("\nĐã lưu metadata tại:")
print(META_PATH)

print("\nHoàn thành quá trình training!")