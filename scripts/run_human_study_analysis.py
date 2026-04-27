from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.append(str(ROOT / "src"))


SPEAKERS = [f"Speaker {label}" for label in "ABCDEFG"]


def normalize_choice(value: str) -> str:
    cleaned = str(value).strip()
    return cleaned


def detect_blocks(header: list[str]) -> list[list[int]]:
    """Split repeated Google Form columns into one block per summary stimulus."""
    lower = [col.lower() for col in header]
    contribution_indices = [
        idx for idx, col in enumerate(lower)
        if "contribute the most" in col
    ]

    blocks: list[list[int]] = []
    for block_idx, start in enumerate(contribution_indices):
        end = contribution_indices[block_idx + 1] if block_idx + 1 < len(contribution_indices) else len(header)
        block_cols = list(range(start, end))
        # Conversation 1 includes an extra "influence" question; ignore it so every
        # summary block is analyzed with the same 5-question structure.
        block_cols = [
            idx for idx in block_cols
            if "influential in shaping the discussion" not in lower[idx]
        ]
        blocks.append(block_cols)

    return blocks


def parse_responses(response_csv: str) -> tuple[list[dict[str, object]], list[str]]:
    response_path = Path(response_csv)
    with response_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = list(csv.reader(handle))

    if not reader:
        raise ValueError(f"No rows found in response file: {response_csv}")

    header = reader[0]
    rows = reader[1:]
    blocks = detect_blocks(header)

    long_rows: list[dict[str, object]] = []
    for participant_index, row in enumerate(rows, start=1):
        timestamp = row[0].strip() if row else ""
        consent = row[1].strip() if len(row) > 1 else ""

        for block_number, block_indices in enumerate(blocks, start=1):
            record = {
                "participant_id": participant_index,
                "timestamp": timestamp,
                "consent": consent,
                "stimulus_block": block_number,
                "contribution_choice": "",
                "knowledge_choice": "",
                "trust_choice": "",
                "balance_rating": "",
                "reasoning": "",
            }

            for idx in block_indices:
                column = header[idx].lower()
                value = row[idx].strip() if idx < len(row) else ""
                if not value:
                    continue

                if "contribute the most" in column:
                    record["contribution_choice"] = normalize_choice(value)
                elif "knowledgable" in column or "knowledgeable" in column:
                    record["knowledge_choice"] = normalize_choice(value)
                elif "trust" in column:
                    record["trust_choice"] = normalize_choice(value)
                elif "balanced" in column:
                    record["balance_rating"] = value
                elif "reasoning" in column:
                    record["reasoning"] = value

            long_rows.append(record)

    return long_rows, header


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise_overall(long_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    contribution = Counter()
    knowledge = Counter()
    trust = Counter()
    balance_values: list[float] = []

    for row in long_rows:
        if row["consent"] and str(row["consent"]).lower() != "yes":
            continue

        if row["contribution_choice"]:
            contribution[str(row["contribution_choice"])] += 1
        if row["knowledge_choice"]:
            knowledge[str(row["knowledge_choice"])] += 1
        if row["trust_choice"]:
            trust[str(row["trust_choice"])] += 1
        if row["balance_rating"]:
            try:
                balance_values.append(float(str(row["balance_rating"])))
            except ValueError:
                pass

    def metric_rows(metric_name: str, counts: Counter) -> list[dict[str, object]]:
        total = sum(counts.values())
        out = []
        for speaker in SPEAKERS:
            count = counts.get(speaker, 0)
            out.append(
                {
                    "metric": metric_name,
                    "speaker": speaker,
                    "count": count,
                    "percentage": round((100.0 * count / total), 2) if total else 0.0,
                }
            )
        return out

    summary_rows = (
        metric_rows("contribution", contribution)
        + metric_rows("knowledge", knowledge)
        + metric_rows("trust", trust)
    )

    overall = {
        "contribution": contribution,
        "knowledge": knowledge,
        "trust": trust,
        "balance_values": balance_values,
    }
    return summary_rows, overall


def summarise_blocks(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    block_ids = sorted({int(row["stimulus_block"]) for row in long_rows})
    rows: list[dict[str, object]] = []

    for block_id in block_ids:
        subset = [row for row in long_rows if int(row["stimulus_block"]) == block_id and str(row["consent"]).lower() == "yes"]
        contribution = Counter(str(row["contribution_choice"]) for row in subset if row["contribution_choice"])
        knowledge = Counter(str(row["knowledge_choice"]) for row in subset if row["knowledge_choice"])
        trust = Counter(str(row["trust_choice"]) for row in subset if row["trust_choice"])
        balance_values = []
        for row in subset:
            if row["balance_rating"]:
                try:
                    balance_values.append(float(str(row["balance_rating"])))
                except ValueError:
                    pass

        top_contribution = contribution.most_common(1)[0] if contribution else ("", 0)
        top_knowledge = knowledge.most_common(1)[0] if knowledge else ("", 0)
        top_trust = trust.most_common(1)[0] if trust else ("", 0)

        rows.append(
            {
                "stimulus_block": block_id,
                "n_responses": len(subset),
                "top_contribution_speaker": top_contribution[0],
                "top_contribution_count": top_contribution[1],
                "top_knowledge_speaker": top_knowledge[0],
                "top_knowledge_count": top_knowledge[1],
                "top_trust_speaker": top_trust[0],
                "top_trust_count": top_trust[1],
                "balance_mean": round(sum(balance_values) / len(balance_values), 3) if balance_values else "",
            }
        )

    return rows


def write_text_summary(path: Path, long_rows: list[dict[str, object]], overall: dict[str, object]) -> None:
    balance_values = overall["balance_values"]
    contribution: Counter = overall["contribution"]
    knowledge: Counter = overall["knowledge"]
    trust: Counter = overall["trust"]

    consenting_rows = [row for row in long_rows if str(row["consent"]).lower() == "yes"]
    participants = len({int(row["participant_id"]) for row in consenting_rows})
    stimuli = len({int(row["stimulus_block"]) for row in consenting_rows})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Human Study Summary (Chroma)\n")
        handle.write("============================\n\n")
        handle.write(f"Participants (consented): {participants}\n")
        handle.write(f"Stimuli blocks: {stimuli}\n")
        handle.write(f"Total block-level response rows: {len(consenting_rows)}\n")
        handle.write(f"Mean balance rating: {round(sum(balance_values) / len(balance_values), 3) if balance_values else 'n/a'}\n\n")

        handle.write("Top contribution choices:\n")
        for speaker, count in contribution.most_common():
            handle.write(f"- {speaker}: {count}\n")

        handle.write("\nTop knowledge choices:\n")
        for speaker, count in knowledge.most_common():
            handle.write(f"- {speaker}: {count}\n")

        handle.write("\nTop trust choices:\n")
        for speaker, count in trust.most_common():
            handle.write(f"- {speaker}: {count}\n")


def plot_choice_summary(overall: dict[str, object], output_path: Path) -> None:
    contribution: Counter = overall["contribution"]
    knowledge: Counter = overall["knowledge"]
    trust: Counter = overall["trust"]

    x = range(len(SPEAKERS))
    width = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar([i - width for i in x], [contribution.get(s, 0) for s in SPEAKERS], width=width, label="Contribution")
    plt.bar(x, [knowledge.get(s, 0) for s in SPEAKERS], width=width, label="Knowledge")
    plt.bar([i + width for i in x], [trust.get(s, 0) for s in SPEAKERS], width=width, label="Trust")
    plt.xticks(list(x), [s.replace("Speaker ", "") for s in SPEAKERS])
    plt.xlabel("Speaker")
    plt.ylabel("Number of selections")
    plt.title("Human Study: Perceived Contribution, Knowledge, and Trust")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_choice_summary_percent(overall: dict[str, object], output_path: Path) -> None:
    contribution: Counter = overall["contribution"]
    knowledge: Counter = overall["knowledge"]
    trust: Counter = overall["trust"]

    totals = {
        "contribution": max(sum(contribution.values()), 1),
        "knowledge": max(sum(knowledge.values()), 1),
        "trust": max(sum(trust.values()), 1),
    }

    x = range(len(SPEAKERS))
    width = 0.25
    contribution_pct = [100.0 * contribution.get(s, 0) / totals["contribution"] for s in SPEAKERS]
    knowledge_pct = [100.0 * knowledge.get(s, 0) / totals["knowledge"] for s in SPEAKERS]
    trust_pct = [100.0 * trust.get(s, 0) / totals["trust"] for s in SPEAKERS]

    plt.figure(figsize=(11, 6))
    plt.bar([i - width for i in x], contribution_pct, width=width, label="Contribution")
    plt.bar(x, knowledge_pct, width=width, label="Knowledge")
    plt.bar([i + width for i in x], trust_pct, width=width, label="Trust")
    plt.xticks(list(x), [s.replace("Speaker ", "") for s in SPEAKERS])
    plt.xlabel("Speaker")
    plt.ylabel("Selection Share (%)")
    plt.title("Human Study: Perceived Contribution, Knowledge, and Trust (Percent)")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_balance_distribution(overall: dict[str, object], output_path: Path) -> None:
    balance_values = overall["balance_values"]
    counts = Counter(int(v) for v in balance_values)
    levels = [1, 2, 3, 4, 5]

    plt.figure(figsize=(8, 5))
    plt.bar(levels, [counts.get(level, 0) for level in levels])
    plt.xlabel("Balance Rating")
    plt.ylabel("Count")
    plt.title("Human Study: Balance Rating Distribution")
    plt.xticks(levels)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_balance_distribution_percent(overall: dict[str, object], output_path: Path) -> None:
    balance_values = overall["balance_values"]
    counts = Counter(int(v) for v in balance_values)
    levels = [1, 2, 3, 4, 5]
    total = max(sum(counts.values()), 1)
    percentages = [100.0 * counts.get(level, 0) / total for level in levels]

    plt.figure(figsize=(8, 5))
    plt.bar(levels, percentages)
    plt.xlabel("Balance Rating")
    plt.ylabel("Share of Ratings (%)")
    plt.title("Human Study: Balance Rating Distribution (Percent)")
    plt.xticks(levels)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_human_study_analysis(response_csv: str, output_dir: str, figures_dir: str) -> None:
    long_rows, header = parse_responses(response_csv)
    summary_rows, overall = summarise_overall(long_rows)
    block_rows = summarise_blocks(long_rows)

    output_path = Path(output_dir)
    figures_path = Path(figures_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    long_csv = output_path / "human_study_long_chroma.csv"
    summary_csv = output_path / "human_study_summary_chroma.csv"
    block_csv = output_path / "human_study_block_summary_chroma.csv"
    summary_txt = output_path / "human_study_summary_chroma.txt"
    balance_csv = output_path / "human_study_balance_distribution_chroma.csv"

    write_csv(
        long_csv,
        long_rows,
        [
            "participant_id",
            "timestamp",
            "consent",
            "stimulus_block",
            "contribution_choice",
            "knowledge_choice",
            "trust_choice",
            "balance_rating",
            "reasoning",
        ],
    )
    write_csv(summary_csv, summary_rows, ["metric", "speaker", "count", "percentage"])
    write_csv(
        block_csv,
        block_rows,
        [
            "stimulus_block",
            "n_responses",
            "top_contribution_speaker",
            "top_contribution_count",
            "top_knowledge_speaker",
            "top_knowledge_count",
            "top_trust_speaker",
            "top_trust_count",
            "balance_mean",
        ],
    )
    balance_counts = Counter(int(v) for v in overall["balance_values"])
    balance_rows = []
    total_balance = max(sum(balance_counts.values()), 1)
    for level in [1, 2, 3, 4, 5]:
        count = balance_counts.get(level, 0)
        balance_rows.append(
            {
                "balance_rating": level,
                "count": count,
                "percentage": round(100.0 * count / total_balance, 2),
            }
        )
    write_csv(balance_csv, balance_rows, ["balance_rating", "count", "percentage"])
    write_text_summary(summary_txt, long_rows, overall)

    plot_choice_summary(overall, figures_path / "human_study_choices_chroma.png")
    plot_choice_summary_percent(overall, figures_path / "human_study_choices_percent_chroma.png")
    plot_balance_distribution(overall, figures_path / "human_study_balance_chroma.png")
    plot_balance_distribution_percent(overall, figures_path / "human_study_balance_percent_chroma.png")

    print(f"Saved long-format responses to: {long_csv}")
    print(f"Saved summary table to: {summary_csv}")
    print(f"Saved block-level summary to: {block_csv}")
    print(f"Saved balance summary to: {balance_csv}")
    print(f"Saved text summary to: {summary_txt}")
    print(f"Saved plots to: {figures_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses",
        default="data/raw/Meeting Summary Survey (Responses) - Form Responses 1.csv",
        help="Path to Google Form CSV export for the human study",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/results",
        help="Directory for generated summary CSV/TXT files",
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/figures",
        help="Directory for generated plots",
    )
    args = parser.parse_args()

    run_human_study_analysis(args.responses, args.output_dir, args.figures_dir)
