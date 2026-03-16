from __future__ import annotations

import re
import pandas as pd


HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "i think", "i guess",
    "sort of", "kind of", "probably", "seems"
}

POLITENESS_WORDS = {
    "please", "thanks", "thank you", "sorry", "appreciate"
}


def count_matches(text: str, phrases: set[str]) -> int:
    text_lower = str(text).lower()
    return sum(1 for phrase in phrases if phrase in text_lower)


def extract_linguistic_features(df: pd.DataFrame, text_col: str = "message") -> pd.DataFrame:
    out = df.copy()

    out[text_col] = out[text_col].fillna("").astype(str)

    out["message_length_chars"] = out[text_col].str.len()
    out["message_length_words"] = out[text_col].str.split().str.len()

    out["has_question"] = out[text_col].str.contains(r"\?", regex=True).astype(int)
    out["has_exclamation"] = out[text_col].str.contains(r"!", regex=False).astype(int)

    out["num_uppercase_chars"] = out[text_col].apply(lambda x: sum(1 for c in x if c.isupper()))
    out["num_digits"] = out[text_col].apply(lambda x: sum(1 for c in x if c.isdigit()))

    out["num_hedges"] = out[text_col].apply(lambda x: count_matches(x, HEDGE_WORDS))
    out["num_politeness"] = out[text_col].apply(lambda x: count_matches(x, POLITENESS_WORDS))

    out["num_first_person_pronouns"] = out[text_col].apply(
        lambda x: len(re.findall(r"\b(i|me|my|mine|we|us|our|ours)\b", x.lower()))
    )

    return out