from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def compute_gini(values: list[float]) -> float:
    arr = np.array(values, dtype=float)

    if len(arr) == 0:
        return 0.0

    if np.all(arr == 0):
        return 0.0

    if np.min(arr) < 0:
        arr = arr - np.min(arr)

    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)

    gini = np.sum((2 * index - n - 1) * arr) / (n * np.sum(arr))
    return float(gini)


def compute_participation_share(df: pd.DataFrame, speaker_col: str = "speaker") -> Dict[str, float]:
    shares = df[speaker_col].value_counts(normalize=True).sort_index()
    return shares.to_dict()


def compute_retrieval_share(
    df_messages: pd.DataFrame,
    df_retrieval: pd.DataFrame,
    speaker_col: str = "speaker",
    msg_id_col: str = "message_uid",
    retrieved_msg_id_col: str = "retrieved_message_uid",
) -> Dict[str, float]:
    speaker_lookup = df_messages.set_index(msg_id_col)[speaker_col].to_dict()

    retrieval_counts = {}
    total = 0

    for retrieved_msg_id in df_retrieval[retrieved_msg_id_col]:
        speaker = speaker_lookup.get(retrieved_msg_id)
        if speaker is None:
            continue
        retrieval_counts[speaker] = retrieval_counts.get(speaker, 0) + 1
        total += 1

    all_speakers = sorted(df_messages[speaker_col].unique())

    if total == 0:
        return {speaker: 0.0 for speaker in all_speakers}

    return {
        speaker: retrieval_counts.get(speaker, 0) / total
        for speaker in all_speakers
    }


def compute_representation_ratio(
    participation_share: Dict[str, float],
    retrieval_share: Dict[str, float],
) -> Dict[str, float]:
    ratios = {}

    all_speakers = sorted(set(participation_share) | set(retrieval_share))

    for speaker in all_speakers:
        p = participation_share.get(speaker, 0.0)
        r = retrieval_share.get(speaker, 0.0)

        if p == 0:
            ratios[speaker] = 0.0
        else:
            ratios[speaker] = r / p

    return ratios


def compute_max_gap(representation_ratio: Dict[str, float]) -> float:
    values = list(representation_ratio.values())
    if not values:
        return 0.0
    return float(max(values) - min(values))


def build_speaker_metrics_table(
    df_messages: pd.DataFrame,
    df_retrieval: pd.DataFrame,
    speaker_col: str = "speaker",
    msg_id_col: str = "message_uid",
    retrieved_msg_id_col: str = "retrieved_message_uid",
) -> pd.DataFrame:
    participation_share = compute_participation_share(
        df_messages,
        speaker_col=speaker_col,
    )

    retrieval_share = compute_retrieval_share(
        df_messages=df_messages,
        df_retrieval=df_retrieval,
        speaker_col=speaker_col,
        msg_id_col=msg_id_col,
        retrieved_msg_id_col=retrieved_msg_id_col,
    )

    representation_ratio = compute_representation_ratio(
        participation_share,
        retrieval_share,
    )

    speakers = sorted(set(participation_share) | set(retrieval_share))

    rows = []
    for speaker in speakers:
        rows.append(
            {
                "speaker": speaker,
                "participation_share": participation_share.get(speaker, 0.0),
                "retrieval_share": retrieval_share.get(speaker, 0.0),
                "representation_ratio": representation_ratio.get(speaker, 0.0),
            }
        )

    result = pd.DataFrame(rows)
    result["gini"] = compute_gini(result["retrieval_share"].tolist())
    result["max_gap"] = compute_max_gap(representation_ratio)

    return result