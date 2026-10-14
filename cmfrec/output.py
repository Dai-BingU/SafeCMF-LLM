"""Parse a JSON object from model output."""
from __future__ import annotations
import json
import re
from typing import Any


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = re.sub('<think>.*?</think>', '', text, flags=re.S).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(value, dict):
                return (value, '')
        except json.JSONDecodeError:
            pass
    return (None, 'no_valid_json_object')
