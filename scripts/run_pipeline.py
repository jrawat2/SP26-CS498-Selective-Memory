from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from generate_plots import main as generate_plots_main  # noqa: E402
from prepare_dataset import prepare_dataset  # noqa: E402
from prepare_human_study import prepare_human_study  # noqa: E402
from run_rag import run_rag  # noqa: E402
from run_retrieval import run_retrieval  # noqa: E402
from run_analysis import run_analysis  # noqa: E402


def run_pipeline(
    input_path: str,
    top_k: int,
    retrievers: list[str],
    generate_plots: bool,
    run_generation: bool,
) -> None:
    processed_path = "data/processed/conversations_clean.csv"
    analysis_dir = "outputs/results"

    prepare_dataset(input_path, processed_path)

    for retriever_name in retrievers:
        retrieval_base_path = "outputs/results/retrieval_results.csv"
        run_retrieval(
            processed_path,
            retrieval_base_path,
            top_k=top_k,
            retriever_name=retriever_name,
        )
        retrieval_path = f"outputs/results/retrieval_results_{retriever_name}.csv"
        run_analysis(processed_path, retrieval_path, analysis_dir)

        if run_generation:
            summaries_path = f"outputs/results/rag_summaries_{retriever_name}.csv"
            trace_path = f"outputs/logs/rag_trace_{retriever_name}.jsonl"
            human_study_path = f"outputs/results/human_study_template_{retriever_name}.csv"
            run_rag(processed_path, retrieval_path, summaries_path, trace_path)
            prepare_human_study(summaries_path, human_study_path)

    if generate_plots and {"tfidf", "dense", "chroma"}.issubset(set(retrievers)):
        generate_plots_main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to conversation CSV")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["tfidf", "dense", "chroma"],
        choices=["tfidf", "dense", "chroma"],
        help="Retriever(s) to run through the full pipeline",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip comparative plot generation",
    )
    parser.add_argument(
        "--run-generation",
        action="store_true",
        help="Generate speaker-citing summaries, trace logs, and human-study templates",
    )
    args = parser.parse_args()

    run_pipeline(
        args.input,
        args.top_k,
        args.retrievers,
        generate_plots=not args.skip_plots,
        run_generation=args.run_generation,
    )
