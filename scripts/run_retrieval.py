from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from selective_memory.retrieval.tfidf_retriever import TfidfRetriever  # noqa: E402
from selective_memory.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from selective_memory.retrieval.chroma_retriever import ChromaRetriever  # noqa: E402


STANDARD_QUERIES = [
    "What is the main topic of discussion?",
    "What are the key points being discussed?",
    "What decisions were made?",
    "What concerns were raised?",
    "What is the recommended approach?",
]


def build_retriever(retriever_name: str):
    retriever_name = retriever_name.lower()

    if retriever_name == "tfidf":
        return TfidfRetriever()
    if retriever_name == "dense":
        return DenseRetriever()
    if retriever_name == "chroma":
        return ChromaRetriever()

    raise ValueError(
        f"Unsupported retriever '{retriever_name}'. Use one of: ['tfidf', 'dense', 'chroma']"
    )


def apply_output_suffix(output_path: str, retriever_name: str) -> Path:
    path = Path(output_path)

    if path.suffix != ".csv":
        raise ValueError("Output path for retrieval results must end with .csv")

    stem = path.stem
    suffix = f"_{retriever_name}"

    if not stem.endswith(suffix):
        path = path.with_name(f"{stem}{suffix}{path.suffix}")

    return path


def run_retrieval(
    input_path: str,
    output_path: str,
    top_k: int,
    retriever_name: str,
) -> None:
    df = pd.read_csv(input_path)

    required_cols = {"conv_id", "message_uid", "message"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Processed dataset is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["conv_id"] = df["conv_id"].astype(str)
    df["message_uid"] = df["message_uid"].astype(str)
    df["message"] = df["message"].fillna("").astype(str)

    rows = []

    for conv_id, conv_df in df.groupby("conv_id"):
        conv_df = conv_df.reset_index(drop=True)

        texts = conv_df["message"].tolist()
        msg_uids = conv_df["message_uid"].tolist()

        retriever = build_retriever(retriever_name)

        if retriever_name.lower() == "chroma":
            retriever.fit(
                texts,
                msg_uids,
                conv_df["speaker"].astype(str).tolist() if "speaker" in conv_df.columns else None,
                conv_df["conv_id"].astype(str).tolist(),
            )
        else:
            retriever.fit(texts)

        for query_id, query_text in enumerate(STANDARD_QUERIES):
            retrieved = retriever.retrieve(query=query_text, top_k=top_k)

            seen_message_uids = set()
            rank_counter = 1

            for doc_index, score in retrieved:
                retrieved_message_uid = msg_uids[doc_index]

                if retrieved_message_uid in seen_message_uids:
                    continue
                seen_message_uids.add(retrieved_message_uid)

                rows.append(
                    {
                        "conv_id": conv_id,
                        "query_id": int(query_id),
                        "query_text": query_text,
                        "retrieved_message_uid": retrieved_message_uid,
                        "rank": rank_counter,
                        "score": float(score),
                        "retriever": retriever_name,
                    }
                )
                rank_counter += 1

                if rank_counter > top_k:
                    break

    out = pd.DataFrame(rows)

    out = out.sort_values(["conv_id", "query_id", "rank"]).reset_index(drop=True)

    out = out.drop_duplicates(
        subset=["conv_id", "query_id", "retrieved_message_uid"],
        keep="first",
    ).reset_index(drop=True)

    out["rank"] = out.groupby(["conv_id", "query_id"]).cumcount() + 1
    out = out[out["rank"] <= top_k].reset_index(drop=True)

    unique_per_query = out.groupby(["conv_id", "query_id"])["retrieved_message_uid"].nunique()

    final_output_path = apply_output_suffix(output_path, retriever_name)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(final_output_path, index=False)

    print("Retriever:", retriever_name)
    print("Min unique retrieved docs per conversation-query:", int(unique_per_query.min()))
    print("Max unique retrieved docs per conversation-query:", int(unique_per_query.max()))
    print("Total unique retrieved message uids:", int(out["retrieved_message_uid"].nunique()))
    print("Total conversation-query groups:", int(unique_per_query.shape[0]))
    print(f"Saved retrieval results to: {final_output_path}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--output", required=True, help="Base output path for retrieval results CSV")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per query")
    parser.add_argument(
        "--retriever",
        type=str,
        default="tfidf",
        choices=["tfidf", "dense", "chroma"],
        help="Retriever type to use",
    )
    args = parser.parse_args()

    run_retrieval(args.input, args.output, args.top_k, args.retriever)