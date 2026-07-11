import difflib

from ..data.noise import strip_diacritics


def classify_edit(original: str, suggestion: str) -> str:
    if strip_diacritics(suggestion) == original and suggestion != original:
        return "diacritic"
    if suggestion.startswith(original) or original.startswith(suggestion):
        return "morphology"
    return "spelling"


def _word_spans(text: str):
    spans = []
    idx = 0
    for word in text.split(" "):
        start = text.index(word, idx) if word else idx
        spans.append((word, start, start + len(word)))
        idx = start + len(word)
    return spans


def compute_edits(original: str, corrected: str) -> list:
    orig_spans = _word_spans(original)
    orig_words = [w for w, _, _ in orig_spans]
    corr_words = corrected.split(" ")
    sm = difflib.SequenceMatcher(a=orig_words, b=corr_words)
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        orig_text = " ".join(orig_words[i1:i2])
        sugg_text = " ".join(corr_words[j1:j2])
        if i1 < len(orig_spans):
            start = orig_spans[i1][1]
            end = orig_spans[i2 - 1][2] if i2 - 1 < len(orig_spans) and i2 > i1 else start
        else:
            start = end = len(original)
        edits.append({
            "start": start,
            "end": end,
            "original": orig_text,
            "suggestion": sugg_text,
            "type": classify_edit(orig_text, sugg_text),
        })
    return edits
