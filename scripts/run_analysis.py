from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from selective_memory.features.linguistic_features import extract_linguistic_features  # noqa: E402
from selective_memory.metrics.fairness_metrics import build_speaker_metrics_table  # noqa: E402
from selective_memory.models.logistic_regression import (  # noqa: E402
    build_message_level_retrieval_labels,
    fit_logistic_regression,
    fit_logistic_regression_style_only,
)


def run_analysis(messages_path: str, retrieval_path: str, output_dir: str) -> None:
    df_messages = pd.read_csv(messages_path)
    df_retrieval = pd.read_csv(retrieval_path)

    required_message_cols = {"msg_id", "speaker", "message"}
    missing_message_cols = required_message_cols - set(df_messages.columns)
    if missing_message_cols:
        raise ValueError(
            f"Messages file is missing required columns: {sorted(missing_message_cols)}"
        )

    required_retrieval_cols = {"retrieved_msg_id"}
    missing_retrieval_cols = required_retrieval_cols - set(df_retrieval.columns)
    if missing_retrieval_cols:
        raise ValueError(
            f"Retrieval file is missing required columns: {sorted(missing_retrieval_cols)}"
        )

    df_messages = df_messages.copy()
    df_retrieval = df_retrieval.copy()

    df_messages["msg_id"] = pd.to_numeric(df_messages["msg_id"], errors="coerce")
    df_retrieval["retrieved_msg_id"] = pd.to_numeric(
        df_retrieval["retrieved_msg_id"], errors="coerce"
    )

    df_messages = df_messages.dropna(subset=["msg_id"]).copy()
    df_retrieval = df_retrieval.dropna(subset=["retrieved_msg_id"]).copy()

    df_messages["msg_id"] = df_messages["msg_id"].astype(int)
    df_retrieval["retrieved_msg_id"] = df_retrieval["retrieved_msg_id"].astype(int)

    # Feature extraction
    df_messages = extract_linguistic_features(df_messages)

    # Binary label: whether each message was retrieved in any standardized query
    df_labeled = build_message_level_retrieval_labels(
        df_messages=df_messages,
        df_retrieval=df_retrieval,
        msg_id_col="msg_id",
        retrieved_msg_id_col="retrieved_msg_id",
    )

    # Speaker-level fairness metrics
    speaker_metrics = build_speaker_metrics_table(
        df_messages=df_messages,
        df_retrieval=df_retrieval,
        speaker_col="speaker",
        msg_id_col="msg_id",
    )

    # Regression 1: style + speaker
    model_full, coef_df_full, regression_summary_full = fit_logistic_regression(df_labeled)

    # Regression 2: style only
    model_style, coef_df_style, regression_summary_style = fit_logistic_regression_style_only(
        df_labeled
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    speaker_metrics_path = output_path / "speaker_metrics.csv"
    message_features_path = output_path / "message_level_features.csv"
    regression_summary_path = output_path / "regression_summary.txt"
    coefficients_full_path = output_path / "regression_coefficients.csv"
    coefficients_style_path = output_path / "regression_coefficients_style_only.csv"

    speaker_metrics.to_csv(speaker_metrics_path, index=False)
    df_labeled.to_csv(message_features_path, index=False)
    coef_df_full.to_csv(coefficients_full_path, index=False)
    coef_df_style.to_csv(coefficients_style_path, index=False)

    retrieved_counts = df_labeled["retrieved"].value_counts().to_dict()
    total_messages = len(df_labeled)
    total_retrieved_unique = int(df_labeled["retrieved"].sum())
    total_retrieval_events = len(df_retrieval)

    with open(regression_summary_path, "w") as f:
        f.write("Logistic regression fitted successfully.\n")
        f.write("Proposal-style retrieval setting: standardized queries per conversation.\n\n")

        f.write(f"Total messages: {total_messages}\n")
        f.write(f"Total unique retrieved messages: {total_retrieved_unique}\n")
        f.write(f"Total retrieval events: {total_retrieval_events}\n")
        f.write(f"Retrieved label distribution: {retrieved_counts}\n\n")

        f.write("=== Model 1: Style + Speaker ===\n")
        f.write(f"Intercept: {regression_summary_full['intercept']:.6f}\n")
        f.write(f"Train accuracy: {regression_summary_full['train_accuracy']:.4f}\n")
        f.write(f"Positive samples: {regression_summary_full['n_positive']}\n")
        f.write(f"Negative samples: {regression_summary_full['n_negative']}\n")
        f.write(
            f"Categorical features used: {regression_summary_full['categorical_features_used']}\n"
        )
        f.write(
            f"Numeric features used: {regression_summary_full['numeric_features_used']}\n\n"
        )

        f.write("=== Model 2: Style Only ===\n")
        f.write(f"Intercept: {regression_summary_style['intercept']:.6f}\n")
        f.write(f"Train accuracy: {regression_summary_style['train_accuracy']:.4f}\n")
        f.write(f"Positive samples: {regression_summary_style['n_positive']}\n")
        f.write(f"Negative samples: {regression_summary_style['n_negative']}\n")
        f.write(
            f"Categorical features used: {regression_summary_style['categorical_features_used']}\n"
        )
        f.write(
            f"Numeric features used: {regression_summary_style['numeric_features_used']}\n\n"
        )

        f.write("Top coefficients saved in:\n")
        f.write("- regression_coefficients.csv\n")
        f.write("- regression_coefficients_style_only.csv\n")

    print(f"Saved speaker metrics to: {speaker_metrics_path}")
    print(f"Saved message-level features to: {message_features_path}")
    print(f"Saved regression summary to: {regression_summary_path}")
    print(f"Saved regression coefficients to: {coefficients_full_path}")
    print(f"Saved style-only regression coefficients to: {coefficients_style_path}")

    print("\nAnalysis diagnostics:")
    print(f"Total messages: {total_messages}")
    print(f"Total unique retrieved messages: {total_retrieved_unique}")
    print(f"Total retrieval events: {total_retrieval_events}")
    print(f"Retrieved label distribution: {retrieved_counts}")

    print("\nSpeaker metrics preview:")
    print(speaker_metrics)

    print("\nTop regression coefficients (style + speaker):")
    print(coef_df_full.head(15))

    print("\nTop regression coefficients (style only):")
    print(coef_df_style.head(15))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", required=True, help="Path to processed CSV")
    parser.add_argument("--retrieval", required=True, help="Path to retrieval results CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    run_analysis(args.messages, args.retrieval, args.output_dir)