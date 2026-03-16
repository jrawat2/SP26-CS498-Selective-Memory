from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from selective_memory.retrieval.tfidf_retriever import TfidfRetriever  # noqa: E402


STANDARD_QUERIES = [
    "What is the main topic of discussion?",
    "What are the key points being discussed?",
    "What decisions were made?",
    "What concerns were raised?",
    "What is the recommended approach?",
]


def run_retrieval(input_path: str, output_path: str, top_k: int) -> None:
    df = pd.read_csv(input_path)

    required_cols = {"conv_id", "msg_id", "message"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Processed dataset is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["conv_id"] = df["conv_id"].astype(str)
    df["message"] = df["message"].fillna("").astype(str)

    rows = []

    for conv_id, conv_df in df.groupby("conv_id"):
        conv_df = conv_df.reset_index(drop=True)

        texts = conv_df["message"].tolist()
        msg_ids = conv_df["msg_id"].tolist()

        retriever = TfidfRetriever()
        retriever.fit(texts)

        for query_id, query_text in enumerate(STANDARD_QUERIES):
            retrieved = retriever.retrieve(query=query_text, top_k=top_k)

            seen_msg_ids = set()
            rank_counter = 1

            for doc_index, score in retrieved:
                retrieved_msg_id = msg_ids[doc_index]

                if retrieved_msg_id in seen_msg_ids:
                    continue
                seen_msg_ids.add(retrieved_msg_id)

                rows.append(
                    {
                        "conv_id": conv_id,
                        "query_id": int(query_id),
                        "query_text": query_text,
                        "retrieved_msg_id": retrieved_msg_id,
                        "rank": rank_counter,
                        "score": float(score),
                        "retriever": "tfidf",
                    }
                )
                rank_counter += 1

                if rank_counter > top_k:
                    break

    out = pd.DataFrame(rows)

    out = out.sort_values(["conv_id", "query_id", "rank"]).reset_index(drop=True)

    out = out.drop_duplicates(
        subset=["conv_id", "query_id", "retrieved_msg_id"],
        keep="first",
    ).reset_index(drop=True)

    out["rank"] = out.groupby(["conv_id", "query_id"]).cumcount() + 1
    out = out[out["rank"] <= top_k].reset_index(drop=True)

    unique_per_query = out.groupby(["conv_id", "query_id"])["retrieved_msg_id"].nunique()
    print("Min unique retrieved docs per conversation-query:", int(unique_per_query.min()))
    print("Max unique retrieved docs per conversation-query:", int(unique_per_query.max()))
    print("Total unique retrieved message ids:", int(out["retrieved_msg_id"].nunique()))
    print("Total conversation-query groups:", int(unique_per_query.shape[0]))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Saved retrieval results to: {output_path}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--output", required=True, help="Path to retrieval results CSV")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per query")
    args = parser.parse_args()

    run_retrieval(args.input, args.output, args.top_k)