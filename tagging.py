from openai import OpenAI
import os
import json
from typing import List, Optional


def build_system_prompt(mapping_labels: List[str], allow_other: bool) -> str:
    base = (
        "You are a strict classification engine. "
        "You receive a single text value from a dataset and must assign it "
        "to exactly one of the allowed labels.\n\n"
        "Rules:\n"
        f"- Allowed labels: {mapping_labels}\n"
        "- Always return a JSON object with the shape: {\"tag\": <one_of_the_labels>}.\n"
    )
    if allow_other:
        base += (
            '- If the value clearly does not fit any label, use "Other".\n'
            '- Do NOT invent new labels.\n'
        )
    else:
        base += (
            "- You must choose one of the allowed labels. "
            "Pick the closest semantic match even if imperfect. "
            "Do NOT use 'Other' or invent new labels.\n"
        )
    return base


def build_user_prompt(
    cell_value: str,
    extra_context: Optional[str],
    column_name: str,
    mapping_labels: List[str],
    allow_other: bool,
) -> str:
    context_str = f"Additional context from the user:\n{extra_context}\n\n" if extra_context else ""
    other_str = '"Other" is allowed.' if allow_other else '"Other" is NOT allowed.'
    return (
        f"You are mapping a value from column '{column_name}'.\n"
        f"{context_str}"
        f"Value to classify: {cell_value!r}\n\n"
        f"Allowed labels: {mapping_labels}\n"
        f"{other_str}\n\n"
        "Return a JSON object, e.g. {\"tag\": \"...\"} with ONLY one of the allowed labels (or 'Other' when allowed)."
    )


def classify_value(
    value: str,
    mapping_labels: List[str],
    column_name: str,
    allow_other: bool,
    extra_context: Optional[str],
    model: str = "gpt-4o-mini",
) -> str:
    # Load key at call time, after .env is loaded
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = build_system_prompt(mapping_labels, allow_other)
    user_prompt = build_user_prompt(value, extra_context, column_name, mapping_labels, allow_other)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    ...