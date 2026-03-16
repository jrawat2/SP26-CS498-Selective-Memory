from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS_NUMERIC = [
    "message_length_chars",
    "message_length_words",
    "has_question",
    "has_exclamation",
    "num_uppercase_chars",
    "num_digits",
    "num_hedges",
    "num_politeness",
    "num_first_person_pronouns",
]

FEATURE_COLUMNS_CATEGORICAL = ["speaker"]
STYLE_ONLY_CATEGORICAL: list[str] = []


def build_message_level_retrieval_labels(
    df_messages: pd.DataFrame,
    df_retrieval: pd.DataFrame,
    msg_id_col: str = "msg_id",
    retrieved_msg_id_col: str = "retrieved_msg_id",
) -> pd.DataFrame:
    out = df_messages.copy()
    retrieved_ids = set(df_retrieval[retrieved_msg_id_col].tolist())
    out["retrieved"] = out[msg_id_col].isin(retrieved_ids).astype(int)
    return out


def _fit_logistic_regression_generic(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    required_cols = numeric_cols + categorical_cols + ["retrieved"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for regression: {missing}")

    work = df.copy()

    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    X = work[numeric_cols + categorical_cols] if categorical_cols else work[numeric_cols]
    y = work["retrieved"]

    if y.nunique() < 2:
        raise ValueError(
            "Logistic regression requires at least 2 classes in 'retrieved'. "
            f"Found only: {sorted(y.unique().tolist())}"
        )

    transformers = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_cols,
        )
    ]

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X, y)

    preprocessor_fitted = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor_fitted.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
            "odds_ratio": np.exp(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    summary = {
        "intercept": float(classifier.intercept_[0]),
        "n_samples": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "train_accuracy": float(model.score(X, y)),
        "categorical_features_used": categorical_cols,
        "numeric_features_used": numeric_cols,
    }

    return model, coef_df, summary


def fit_logistic_regression(df: pd.DataFrame):
    """
    Main regression model:
    linguistic/style features + speaker identity
    """
    return _fit_logistic_regression_generic(
        df=df,
        numeric_cols=FEATURE_COLUMNS_NUMERIC,
        categorical_cols=FEATURE_COLUMNS_CATEGORICAL,
    )


def fit_logistic_regression_style_only(df: pd.DataFrame):
    """
    Secondary regression model:
    linguistic/style features only, without speaker identity
    """
    return _fit_logistic_regression_generic(
        df=df,
        numeric_cols=FEATURE_COLUMNS_NUMERIC,
        categorical_cols=STYLE_ONLY_CATEGORICAL,
    )