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

ASSERTIVE_WORDS = {
    "definitely", "clearly", "must", "certainly", "absolutely", "always",
    "obviously", "the best option", "recommend", "should"
}

MODAL_WORDS = {
    "can", "could", "may", "might", "must", "should", "would", "will"
}

META_CONVERSATION_PHRASES = {
    "quick question", "small question", "just to check", "before we move on",
    "can we clarify", "to be clear", "team:", "let's"
}

QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "can", "could",
    "should", "would", "do", "does", "did", "is", "are"
)


def count_matches(text: str, phrases: set[str]) -> int:
    text_lower = str(text).lower()
    count = 0

    for phrase in phrases:
        if " " in phrase:
            count += text_lower.count(phrase)
        else:
            count += len(re.findall(rf"\b{re.escape(phrase)}\b", text_lower))

    return count


def count_sentences(text: str) -> int:
    sentences = re.findall(r"[^.!?]+[.!?]?", str(text))
    non_empty = [sentence for sentence in sentences if sentence.strip()]
    return max(len(non_empty), 1)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text)))


def count_question_sentences(text: str) -> int:
    text_lower = str(text).lower().strip()
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text_lower) if sentence.strip()]
    total = 0

    for sentence in sentences:
        if sentence.endswith("?"):
            total += 1
            continue
        if sentence.startswith(QUESTION_STARTERS):
            total += 1

    return total


def extract_linguistic_features(df: pd.DataFrame, text_col: str = "message") -> pd.DataFrame:
    out = df.copy()

    out[text_col] = out[text_col].fillna("").astype(str)

    out["message_length_chars"] = out[text_col].str.len()
    out["message_length_words"] = out[text_col].apply(count_words)
    out["num_sentences"] = out[text_col].apply(count_sentences)
    out["avg_word_length"] = out[text_col].apply(
        lambda x: (
            sum(len(word) for word in re.findall(r"\b\w+\b", x)) / max(count_words(x), 1)
        )
    )
    out["words_per_sentence"] = out["message_length_words"] / out["num_sentences"].clip(lower=1)

    out["has_question"] = out[text_col].str.contains(r"\?", regex=True).astype(int)
    out["has_exclamation"] = out[text_col].str.contains(r"!", regex=False).astype(int)
    out["num_questions"] = out[text_col].apply(count_question_sentences)
    out["question_ratio"] = out["num_questions"] / out["num_sentences"].clip(lower=1)
    out["is_interrogative"] = (out["num_questions"] > 0).astype(int)
    out["is_declarative"] = (out["num_questions"] == 0).astype(int)

    out["num_uppercase_chars"] = out[text_col].apply(lambda x: sum(1 for c in x if c.isupper()))
    out["num_digits"] = out[text_col].apply(lambda x: sum(1 for c in x if c.isdigit()))

    out["num_hedges"] = out[text_col].apply(lambda x: count_matches(x, HEDGE_WORDS))
    out["hedge_ratio"] = out["num_hedges"] / out["message_length_words"].clip(lower=1)
    out["num_politeness"] = out[text_col].apply(lambda x: count_matches(x, POLITENESS_WORDS))
    out["num_assertive"] = out[text_col].apply(lambda x: count_matches(x, ASSERTIVE_WORDS))
    out["assertive_ratio"] = out["num_assertive"] / out["message_length_words"].clip(lower=1)
    out["num_modal_verbs"] = out[text_col].apply(lambda x: count_matches(x, MODAL_WORDS))
    out["modal_ratio"] = out["num_modal_verbs"] / out["message_length_words"].clip(lower=1)
    out["num_meta_conversation"] = out[text_col].apply(
        lambda x: count_matches(x, META_CONVERSATION_PHRASES)
    )

    out["num_first_person_pronouns"] = out[text_col].apply(
        lambda x: len(re.findall(r"\b(i|me|my|mine|we|us|our|ours)\b", x.lower()))
    )
    out["confidence_score"] = out["num_assertive"] - out["num_hedges"]

    return out
