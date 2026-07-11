import gzip
from collections import defaultdict
from pathlib import Path


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
