# tagging.py

import json
import os
from typing import List, Optional

from openai import OpenAI


def _get_client(api_key: str) -> OpenAI:
    """Create an OpenAI client using the provided API key."""
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Pass it from app.py (env or Streamlit secrets)."
        )
    return OpenAI(api_key=api_key)


def build_system_prompt(mapping_labels: List[str], allow_other: bool) -> str:
    base = (
        "You are a semantic classification engine that performs fuzzy matching. "
        "You receive a single text value from a dataset and must assign it "
        "to exactly one of the allowed labels using semantic understanding.\n\n"
        "Rules:\n"
        f"- Allowed labels: {mapping_labels}\n"
        "- Always return a JSON object with the shape: {\"tag\": <one_of_the_labels>}.\n"
        "- Use semantic matching: consider synonyms, related terms, variations, and conceptual relationships.\n"
        "- Be generous with matching - prefer mapping to the closest label rather than defaulting to 'Other'.\n"
    )
    if allow_other:
        base += (
            '- Only use "Other" as a LAST RESORT when the value has NO semantic relationship to any label.\n'
            '- Do NOT use "Other" for synonyms, variations, or related concepts - find the best match instead.\n'
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
    if allow_other:
        other_str = (
            '"Other" is allowed ONLY as a last resort when there is truly no semantic match. '
            'Prefer matching to the closest label using synonyms, variations, or related concepts.'
        )
    else:
        other_str = '"Other" is NOT allowed. You must choose one of the allowed labels.'
    return (
        f"You are mapping a value from column '{column_name}'.\n"
        f"{context_str}"
        f"Value to classify: {cell_value!r}\n\n"
        f"Allowed labels: {mapping_labels}\n"
        f"{other_str}\n\n"
        "Use semantic/fuzzy matching to find the best label. Consider synonyms, related terms, "
        "and conceptual relationships. Return a JSON object, e.g. {{\"tag\": \"...\"}} with "
        "ONLY one of the allowed labels (or 'Other' only if truly no match exists)."
    )


def classify_value(
    value: str,
    mapping_labels: List[str],
    column_name: str,
    allow_other: bool,
    extra_context: Optional[str],
    api_key: str,  # <-- this must be here
    model: str = "gpt-4o",
) -> str:
    """
    Classify a single value into one of mapping_labels (or 'Other' if allow_other=True).
    Returns the label as a plain string.
    """
    client = _get_client(api_key)

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

    try:
        data = json.loads(content)
        tag = data.get("tag")
        if tag not in mapping_labels and not (allow_other and tag == "Other"):
            # Fallback: if the model misbehaves, just default to the first label
            return mapping_labels[0]
        return tag
    except Exception:
        # If parsing fails, just default to first label or 'Other' as last resort
        return "Other" if allow_other else mapping_labels[0]