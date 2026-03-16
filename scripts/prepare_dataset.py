from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def prepare_dataset(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    print("Columns found:", list(df.columns))

    # map possible dataset column names to pipeline schema
    rename_map = {
        "speaker_id": "speaker",
        "author": "speaker",
        "user": "speaker",
        "text": "message",
        "content": "message",
    }

    df = df.rename(columns=rename_map)

    # check required columns
    if "speaker" not in df.columns or "message" not in df.columns:
        raise ValueError(
            f"Input CSV must contain speaker/message columns. Found: {list(df.columns)}"
        )

    df = df.copy()

    df["speaker"] = df["speaker"].astype(str).str.strip()
    df["message"] = df["message"].fillna("").astype(str).str.strip()

    # remove empty rows
    df = df[df["speaker"] != ""].reset_index(drop=True)
    df = df[df["message"] != ""].reset_index(drop=True)

    # create IDs if missing
    if "msg_id" not in df.columns:
        df["msg_id"] = range(len(df))

    if "conv_id" not in df.columns:
        df["conv_id"] = 0

    df["conv_id"] = df["conv_id"].astype(str)
    df["msg_id"] = df["msg_id"].astype(str)

    # create globally unique message ID
    df["message_uid"] = df["conv_id"] + "_" + df["msg_id"]

    # reorder columns
    cols = ["message_uid", "msg_id", "conv_id", "speaker", "message"] + [
        c for c in df.columns if c not in {"message_uid", "msg_id", "conv_id", "speaker", "message"}
    ]
    df = df[cols]

    # create output folder
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved cleaned dataset to: {output_path}")
    print(f"Rows: {len(df)}")
    print("Example message_uid:", df["message_uid"].iloc[0] if len(df) > 0 else "N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Path to save cleaned CSV")
    args = parser.parse_args()

    prepare_dataset(args.input, args.output)