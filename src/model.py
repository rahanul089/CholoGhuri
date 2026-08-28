"""
Hybrid ensemble model (Random Forest + Extra Trees + Gradient
Boosting + XGBoost) with a soft-voting classifier, wrapped in a
single sklearn Pipeline so it can be trained and cached inside
Streamlit in a couple of seconds.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False

from .data_processing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

WEIGHT_CONFIGS = {
    "Equal Weighting": [1, 1, 1, 1],
    "Favoring Forests": [3, 3, 1, 1],
    "Favoring Boosting": [1, 1, 3, 3],
    "Balanced": [2, 2, 1, 3],
}


def _make_preprocessor(cat_features, num_features):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
            ]), num_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_features),
        ]
    )


def train_ensemble(df, feature_cols, cat_features, num_features, random_state=42):
    """Train the hybrid ensemble and return everything needed for
    inference + explainability + a small performance report."""
    X = df[feature_cols]
    y = df["would_recommend_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = _make_preprocessor(cat_features, num_features)

    base_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=random_state, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=random_state, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            min_samples_split=5, min_samples_leaf=2, random_state=random_state,
        ),
    }
    if HAS_XGB:
        base_models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, eval_metric="logloss",
        )

    base_results = {}
    for name, mdl in base_models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", mdl)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        base_results[name] = {
            "pipeline": pipe,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "auc": roc_auc_score(y_test, y_prob),
        }

    # weighted soft-voting ensemble search
    ensemble_results = {}
    names = list(base_models.keys())
    for cfg_name, weights in WEIGHT_CONFIGS.items():
        estimators = [(n[:3].lower(), base_results[n]["pipeline"].named_steps["classifier"]) for n in names]
        voting = VotingClassifier(estimators=estimators, voting="soft", weights=weights[: len(estimators)], n_jobs=-1)
        pipe = Pipeline([("preprocessor", preprocessor), ("ensemble", voting)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        ensemble_results[cfg_name] = {
            "pipeline": pipe,
            "weights": weights[: len(estimators)],
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "auc": roc_auc_score(y_test, y_prob),
            "cv_mean": cv.mean(),
            "cv_std": cv.std(),
        }

    best_name = max(ensemble_results, key=lambda k: ensemble_results[k]["accuracy"])
    best_pipeline = ensemble_results[best_name]["pipeline"]

    # fit the fitted preprocessor + a plain RF for SHAP (voting classifier
    # itself isn't directly SHAP-friendly, so we explain with one of its members)
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    rf_for_shap = base_results["Random Forest"]["pipeline"].named_steps["classifier"]

    cat_names = list(
        fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_features)
    )
    all_feature_names = num_features + cat_names

    return {
        "best_name": best_name,
        "pipeline": best_pipeline,
        "preprocessor": fitted_preprocessor,
        "base_results": base_results,
        "ensemble_results": ensemble_results,
        "rf_for_shap": rf_for_shap,
        "feature_names": all_feature_names,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
    }


def predict(user_row_df, feature_cols, pipeline):
    row = user_row_df.copy()
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols]
    pred = pipeline.predict(row)[0]
    prob = pipeline.predict_proba(row)[:, 1][0]
    return int(pred), float(prob)


def explain_prediction(user_row_df, feature_cols, model_bundle, top_n=6):
    """Return the top contributing features for a single prediction
    using a SHAP TreeExplainer on the Random Forest ensemble member.
    Falls back to global feature importances if SHAP isn't available."""
    row = user_row_df.copy()
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols]

    preprocessor = model_bundle["preprocessor"]
    X_transformed = preprocessor.transform(row)
    feature_names = model_bundle["feature_names"]

    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model_bundle["rf_for_shap"])
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif len(np.shape(shap_values)) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]
            order = np.argsort(np.abs(sv))[::-1][:top_n]
            return [(feature_names[i], float(sv[i])) for i in order]
        except Exception:
            pass

    # fallback: global RF feature importances (sign-agnostic)
    importances = model_bundle["rf_for_shap"].feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in order]
