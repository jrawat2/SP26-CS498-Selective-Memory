from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")


def load_metrics(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def plot_representation_ratio_comparison(
    df_tfidf: pd.DataFrame,
    df_dense: pd.DataFrame,
) -> None:
    merged = df_tfidf[["speaker", "representation_ratio"]].merge(
        df_dense[["speaker", "representation_ratio"]],
        on="speaker",
        suffixes=("_tfidf", "_dense"),
    )

    x = range(len(merged))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], merged["representation_ratio_tfidf"], width=width, label="TF-IDF")
    plt.bar([i + width / 2 for i in x], merged["representation_ratio_dense"], width=width, label="Dense")

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xticks(list(x), merged["speaker"])
    plt.xlabel("Speaker")
    plt.ylabel("Representation Ratio")
    plt.title("Speaker Representation Ratio: TF-IDF vs Dense Retrieval")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "speaker_representation_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_participation_vs_retrieval(
    df: pd.DataFrame,
    retriever_name: str,
) -> None:
    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], df["participation_share"], width=width, label="Participation Share")
    plt.bar([i + width / 2 for i in x], df["retrieval_share"], width=width, label="Retrieval Share")

    plt.xticks(list(x), df["speaker"])
    plt.xlabel("Speaker")
    plt.ylabel("Share")
    plt.title(f"Participation vs Retrieval Share ({retriever_name})")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / f"participation_vs_retrieval_{retriever_name.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_gini_comparison(
    df_tfidf: pd.DataFrame,
    df_dense: pd.DataFrame,
) -> None:
    gini_tfidf = float(df_tfidf["gini"].iloc[0])
    gini_dense = float(df_dense["gini"].iloc[0])

    methods = ["TF-IDF", "Dense"]
    gini_values = [gini_tfidf, gini_dense]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, gini_values)
    plt.ylabel("Gini Coefficient")
    plt.title("Retrieval Inequality Comparison")
    plt.tight_layout()

    output_path = FIGURES_DIR / "gini_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def save_summary_table(
    df_tfidf: pd.DataFrame,
    df_dense: pd.DataFrame,
) -> None:
    summary = pd.DataFrame(
        {
            "retriever": ["tfidf", "dense"],
            "gini": [float(df_tfidf["gini"].iloc[0]), float(df_dense["gini"].iloc[0])],
            "max_gap": [float(df_tfidf["max_gap"].iloc[0]), float(df_dense["max_gap"].iloc[0])],
        }
    )

    output_path = RESULTS_DIR / "retriever_comparison_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def plot_retrieval_histogram(df_metrics: pd.DataFrame, retriever_name: str):
    """
    Histogram showing number of retrieved messages per speaker
    """

    # total retrieval events for this retriever
    total_retrievals = df_metrics["retrieval_share"].sum()

    # convert share → counts
    df_metrics["retrieval_count"] = (
        df_metrics["retrieval_share"] * total_retrievals
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        df_metrics["speaker"],
        df_metrics["retrieval_count"]
    )

    plt.xlabel("Speaker")
    plt.ylabel("Retrieved Messages")
    plt.title(f"Retrieved Message Distribution ({retriever_name})")

    plt.tight_layout()

    output_path = FIGURES_DIR / f"retrieval_histogram_{retriever_name.lower()}.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_tfidf = load_metrics("speaker_metrics_tfidf.csv")
    df_dense = load_metrics("speaker_metrics_dense.csv")

    plot_representation_ratio_comparison(df_tfidf, df_dense)
    plot_participation_vs_retrieval(df_tfidf, "TFIDF")
    plot_participation_vs_retrieval(df_dense, "Dense")
    plot_gini_comparison(df_tfidf, df_dense)
    save_summary_table(df_tfidf, df_dense)
    plot_retrieval_histogram(df_tfidf, "TFIDF")
    plot_retrieval_histogram(df_dense, "Dense")

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()