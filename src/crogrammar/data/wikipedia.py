import regex as re

_MARKUP = ("==", "[[", "]]", "{{", "}}", "|")
_LETTER_RE = re.compile(r"\p{L}", re.UNICODE)
_DIGIT_RE = re.compile(r"\d")


def is_clean_sentence(text: str) -> bool:
    text = text.strip()
    if len(text) < 20 or len(text) > 200:
        return False
    if any(m in text for m in _MARKUP):
        return False
    letters = len(_LETTER_RE.findall(text))
    if letters < len(text) * 0.5:
        return False
    digits = len(_DIGIT_RE.findall(text))
    if digits > len(text) * 0.2:
        return False
    if "http" in text.lower():
        return False
    return True


def load_wiki_sentences(limit=None):
    from datasets import load_dataset
    from ..inference.segment import split_sentences
    ds = load_dataset("wikimedia/wikipedia", "20231101.hr", split="train",
                      streaming=True)
    seen = set()
    count = 0
    for article in ds:
        for sent in split_sentences(article["text"]):
            if not is_clean_sentence(sent):
                continue
            if sent in seen:
                continue
            seen.add(sent)
            yield sent
            count += 1
            if limit and count >= limit:
                return
