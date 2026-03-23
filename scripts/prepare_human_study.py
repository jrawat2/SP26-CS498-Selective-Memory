from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RATING_COLUMNS = [
    "contribution_rating",
    "knowledge_rating",
    "trust_rating",
    "leader_choice",
    "notes",
]


def prepare_human_study(summaries_path: str, output_path: str) -> None:
    df = pd.read_csv(summaries_path)
    required_cols = {"conv_id", "query_id", "retriever", "summary"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Summary file is missing required columns: {sorted(missing)}")

    out = df.copy()
    out.insert(0, "stimulus_id", range(1, len(out) + 1))

    for column in RATING_COLUMNS:
        out[column] = ""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved human-study template to: {output_path}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", required=True, help="Path to generated summary CSV")
    parser.add_argument("--output", required=True, help="Path to save annotation template CSV")
    args = parser.parse_args()

    prepare_human_study(args.summaries, args.output)
