from __future__ import annotations

import re

from .types import NumericMention


_NUMERIC_TOKEN_RE = re.compile(
    r"""
    (?:
        (?P<lead>\.)\d+(?:\.\d+)?
        (?:st|nd|rd|th)?
        (?:%|bp|bps|k|K|m|M|b|B)?
        \b
    )
    |
    (?:
        [+-]?
        (?:
            \d{1,3}(?:,\d{3})+(?:\.\d+)?
            |
            \d+(?:\.\d+)?
        )
        (?:st|nd|rd|th)?
        (?:%|bp|bps|k|K|m|M|b|B)?
        \b
    )
    """,
    flags=re.VERBOSE,
)


def _clean_numeric_string(token: str) -> str:
    cleaned = token.strip()
    cleaned = re.sub(r"(st|nd|rd|th)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", "")
    if cleaned.startswith("."):
        cleaned = "0" + cleaned
    if cleaned.startswith("-."):
        cleaned = cleaned.replace("-.", "-0.", 1)
    if cleaned.startswith("+."):
        cleaned = cleaned.replace("+.", "+0.", 1)
    cleaned = re.sub(r"(%|bp|bps|k|K|m|M|b|B)\b", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def extract_numeric_mentions(sentence: str) -> list[NumericMention]:
    mentions: list[NumericMention] = []

    for match in _NUMERIC_TOKEN_RE.finditer(sentence):
        token = match.group(0)
        cleaned = _clean_numeric_string(token)
        cleaned = re.sub(r"[^\d\.\-\+]", "", cleaned)
        if cleaned.count(".") > 1:
            parts = cleaned.split(".")
            cleaned = parts[0] + "." + "".join(parts[1:])

        try:
            value = float(cleaned)
        except ValueError:
            continue

        start, end = match.span()
        mentions.append(
            NumericMention(
                text=token,
                value=value,
                start=start,
                end=end,
            )
        )

    return mentions
