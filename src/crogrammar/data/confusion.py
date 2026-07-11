import gzip
from collections import defaultdict
from pathlib import Path

from .noise import strip_diacritics


def build_confusion_set(rows, min_freq: int = 1) -> dict:
    counts = defaultdict(lambda: defaultdict(int))
    for _date, err, corr in rows:
        if err == corr:
            continue
        counts[corr][err] += 1
    result = {}
    for corr, errs in counts.items():
        filtered = [(e, c) for e, c in errs.items() if c >= min_freq]
        if not filtered:
            continue
        filtered.sort(key=lambda x: (-x[1], x[0]))
        result[corr] = [e for e, _ in filtered]
    return result


def read_ispravime_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                yield parts[0], parts[1], parts[2]


def load_confusion_from_dir(raw_dir, min_freq: int = 2) -> dict:
    rows = []
    for gz in sorted(Path(raw_dir).glob("ispravime_*.gz")):
        rows.extend(read_ispravime_gz(gz))
    return build_confusion_set(rows, min_freq=min_freq)


def read_hunspell_dic(text: str):
    lines = text.splitlines()
    if lines and lines[0].strip().isdigit():
        lines = lines[1:]
    for line in lines:
        word = line.split("/", 1)[0].strip()
        if word:
            yield word


def build_confusion_from_wordlist(words) -> dict:
    result = {}
    for word in words:
        stripped = strip_diacritics(word)
        if stripped != word:
            result[word] = [stripped]
    return result


def load_confusion_from_dic(path) -> dict:
    with open(path, encoding="utf-8") as f:
        words = list(read_hunspell_dic(f.read()))
    return build_confusion_from_wordlist(words)
