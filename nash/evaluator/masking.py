from __future__ import annotations

import re

from .extraction import extract_numeric_mentions
from .types import NumericMention


_UNIT_SUFFIX_RE = re.compile(r"(%|bp|bps|k|K|m|M|b|B)\b", flags=re.IGNORECASE)


def mask_numbers(
    sentence: str,
    mentions: list[NumericMention] | None = None,
    mask_token: str = "[NUM]",
) -> str:
    mentions = extract_numeric_mentions(sentence) if mentions is None else mentions
    parts: list[str] = []
    last = 0
    for mention in sorted(mentions, key=lambda item: item.start):
        parts.append(sentence[last : mention.start])
        suffix = _UNIT_SUFFIX_RE.match(sentence[mention.end :])
        if suffix:
            parts.append(mask_token + suffix.group(0))
            last = mention.end + len(suffix.group(0))
        else:
            parts.append(mask_token)
            last = mention.end
    parts.append(sentence[last:])
    return "".join(parts)
