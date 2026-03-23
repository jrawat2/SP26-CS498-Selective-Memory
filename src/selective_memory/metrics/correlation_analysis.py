from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "message_length_chars",
    "message_length_words",
    "num_sentences",
    "avg_word_length",
    "words_per_sentence",
    "has_question",
    "num_questions",
    "question_ratio",
    "num_uppercase_chars",
    "num_digits",
    "num_hedges",
    "hedge_ratio",
    "num_politeness",
    "num_assertive",
    "assertive_ratio",
    "num_modal_verbs",
    "modal_ratio",
    "num_meta_conversation",
    "num_first_person_pronouns",
    "confidence_score",
]


def build_speaker_feature_summary(
    df_labeled: pd.DataFrame,
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    speaker_col: str = "speaker",
    retrieved_col: str = "retrieved",
) -> pd.DataFrame:
    feature_columns = [column for column in feature_columns if column in df_labeled.columns]

    aggregations = {column: "mean" for column in feature_columns}
    aggregations[retrieved_col] = "mean"

    summary = (
        df_labeled.groupby(speaker_col, dropna=False)
        .agg(aggregations)
        .reset_index()
        .rename(columns={retrieved_col: "retrieval_rate"})
        .sort_values(speaker_col)
        .reset_index(drop=True)
    )

    return summary


def compute_feature_retrieval_correlations(
    df_labeled: pd.DataFrame,
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    retrieved_col: str = "retrieved",
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []

    for column in feature_columns:
        if column not in df_labeled.columns:
            continue

        pair = df_labeled[[column, retrieved_col]].copy()
        pair[column] = pd.to_numeric(pair[column], errors="coerce")
        pair[retrieved_col] = pd.to_numeric(pair[retrieved_col], errors="coerce")
        pair = pair.dropna()

        if len(pair) < 2:
            correlation = np.nan
        elif pair[column].nunique() < 2 or pair[retrieved_col].nunique() < 2:
            correlation = np.nan
        else:
            correlation = float(pair[column].corr(pair[retrieved_col], method="pearson"))

        rows.append(
            {
                "feature": column,
                "pearson_correlation": correlation,
                "n_pairs": int(len(pair)),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["abs_pearson_correlation"] = result["pearson_correlation"].abs()
    return result.sort_values(
        ["abs_pearson_correlation", "feature"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
