import regex as re

_SENT_RE = re.compile(r"[^.!?]+[.!?]+|\S[^.!?]*$", re.UNICODE)


def split_sentences(text: str) -> list:
    if not text or not text.strip():
        return []
    return [m.group(0).strip() for m in _SENT_RE.finditer(text) if m.group(0).strip()]
