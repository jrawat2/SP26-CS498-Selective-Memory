from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from selective_memory.rag.pipeline import RAGGenerationResult, RAGPipeline  # noqa: E402


def run_rag(messages_path: str, retrieval_path: str, output_path: str, trace_path: str) -> None:
    df_messages = pd.read_csv(messages_path)
    df_retrieval = pd.read_csv(retrieval_path)

    required_messages = {"message_uid", "speaker", "message", "conv_id"}
    required_retrieval = {"conv_id", "query_id", "query_text", "retrieved_message_uid", "rank", "retriever"}

    missing_messages = required_messages - set(df_messages.columns)
    missing_retrieval = required_retrieval - set(df_retrieval.columns)

    if missing_messages:
        raise ValueError(f"Messages file is missing required columns: {sorted(missing_messages)}")
    if missing_retrieval:
        raise ValueError(f"Retrieval file is missing required columns: {sorted(missing_retrieval)}")

    merged = df_retrieval.merge(
        df_messages[["message_uid", "speaker", "message"]],
        left_on="retrieved_message_uid",
        right_on="message_uid",
        how="left",
    )

    pipeline = RAGPipeline()
    rows: list[dict[str, str | int]] = []

    trace_file = Path(trace_path)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text("", encoding="utf-8")

    for (conv_id, query_id, query_text), group in merged.groupby(["conv_id", "query_id", "query_text"]):
        retriever_values = group["retriever"].dropna().astype(str).unique().tolist()
        retriever_name = retriever_values[0] if retriever_values else "unknown"

        summary, backend, prompt_text = pipeline.generate_summary(query_text, group)
        context = "\n".join(
            [
                f"Speaker {row.speaker}: {row.message}"
                for row in group.sort_values("rank").itertuples()
            ]
        )

        result = RAGGenerationResult(
            conv_id=str(conv_id),
            query_id=int(query_id),
            query_text=str(query_text),
            retriever=str(retriever_name),
            summary=summary,
            generation_backend=backend,
            prompt_text=prompt_text,
            retrieved_context=context,
        )
        pipeline.trace_result(trace_file, result)

        rows.append(
            {
                "conv_id": str(conv_id),
                "query_id": int(query_id),
                "query_text": str(query_text),
                "retriever": str(retriever_name),
                "generation_backend": backend,
                "summary": summary,
            }
        )

    output = pd.DataFrame(rows).sort_values(["conv_id", "query_id"]).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Saved generated summaries to: {output_path}")
    print(f"Saved trace log to: {trace_path}")
    print(json.dumps({"num_summaries": len(output), "generation_backends": output['generation_backend'].value_counts().to_dict()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", required=True, help="Path to processed conversation CSV")
    parser.add_argument("--retrieval", required=True, help="Path to retrieval results CSV")
    parser.add_argument("--output", required=True, help="Path to save summary CSV")
    parser.add_argument(
        "--trace-output",
        required=True,
        help="Path to save JSONL trace records",
    )
    args = parser.parse_args()

    run_rag(args.messages, args.retrieval, args.output, args.trace_output)
