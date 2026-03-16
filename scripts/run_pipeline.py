from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from run_retrieval import run_retrieval  # noqa: E402
from run_analysis import run_analysis  # noqa: E402


def run_pipeline(input_path: str, top_k: int) -> None:
    processed_path = "data/processed/conversations_clean.csv"
    retrieval_path = "outputs/results/retrieval_results.csv"
    analysis_dir = "outputs/results"

    # just copy/standardize with prepare_dataset logic beforehand if needed
    import pandas as pd

    df = pd.read_csv(input_path)
    if "msg_id" not in df.columns:
        df["msg_id"] = range(len(df))
    if "conv_id" not in df.columns:
        df["conv_id"] = 0

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    run_retrieval(processed_path, retrieval_path, top_k=top_k)
    run_analysis(processed_path, retrieval_path, analysis_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to conversation CSV")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    run_pipeline(args.input, args.top_k)