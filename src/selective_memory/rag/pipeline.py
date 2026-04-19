from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


DEFAULT_PROMPT_TEMPLATE = """You are summarizing a group conversation for research auditing.

Question: {query}

Retrieved context:
{context}

Write a concise summary that answers the question and explicitly cites speakers by name when referencing their contributions.
"""


@dataclass
class RAGGenerationResult:
    conv_id: str
    query_id: int
    query_text: str
    retriever: str
    summary: str
    generation_backend: str
    prompt_text: str
    retrieved_context: str


def build_speaker_citation_context(df_retrieved_messages: pd.DataFrame) -> str:
    lines = []

    for _, row in df_retrieved_messages.sort_values("rank").iterrows():
        speaker = str(row.get("speaker", "Unknown"))
        message = str(row.get("message", "")).strip()
        rank = int(row.get("rank", 0))
        score = row.get("score")
        score_text = f"{float(score):.4f}" if score is not None else "n/a"
        lines.append(f"[Rank {rank} | Score {score_text}] Speaker {speaker}: {message}")

    return "\n".join(lines)


class RAGPipeline:
    def __init__(
        self,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        anthropic_model: str = "claude-sonnet-4-6",
    ) -> None:
        self.prompt_template = prompt_template
        self.anthropic_model = anthropic_model

    def build_prompt(self, query: str, context: str) -> str:
        return self.prompt_template.format(query=query, context=context)

    def generate_summary(self, query: str, df_retrieved_messages: pd.DataFrame) -> tuple[str, str, str]:
        context = build_speaker_citation_context(df_retrieved_messages)
        prompt = self.build_prompt(query, context)

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            return self._generate_with_anthropic_api(prompt)

        summary = self._generate_extractive_summary(query, df_retrieved_messages)
        return summary, "extractive_fallback", prompt

    def _generate_with_anthropic_api(self, prompt: str) -> tuple[str, str, str]:
        import anthropic

        api_key = os.environ["ANTHROPIC_API_KEY"]
        model = os.getenv("ANTHROPIC_MODEL", self.anthropic_model)
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        parts: list[str] = []
        for block in message.content:
            if block.type == "text":
                parts.append(block.text)
        text = "".join(parts).strip()
        return text, "anthropic_claude", prompt

    def _generate_extractive_summary(
        self,
        query: str,
        df_retrieved_messages: pd.DataFrame,
    ) -> str:
        bullet_points = []
        for _, row in df_retrieved_messages.sort_values("rank").head(5).iterrows():
            speaker = str(row.get("speaker", "Unknown"))
            message = str(row.get("message", "")).strip()
            bullet_points.append(f"Speaker {speaker} said: {message}")

        joined_points = " ".join(bullet_points)
        return f"For the question '{query}', the retrieved evidence suggests: {joined_points}"

    def trace_result(
        self,
        output_path: str | Path,
        result: RAGGenerationResult,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result), ensure_ascii=True) + "\n")
