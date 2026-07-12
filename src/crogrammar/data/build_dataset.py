import json
import random
from pathlib import Path

from .noise import corrupt_sentence


def make_pairs(clean_sentences, confusion, seed: int, variants: int = 1, real_words=None):
    pairs = []
    for i, tgt in enumerate(clean_sentences):
        for v in range(variants):
            src = corrupt_sentence(tgt, confusion, seed=seed + i * 1000 + v,
                                   real_words=real_words)
            pairs.append({"src": src, "tgt": tgt})
    return pairs


def split_pairs(pairs, dev_frac: float, test_frac: float, seed: int):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_dev = int(n * dev_frac)
    n_test = int(n * test_frac)
    dev = shuffled[:n_dev]
    test = shuffled[n_dev:n_dev + n_test]
    train = shuffled[n_dev + n_test:]
    return train, dev, test


def write_jsonl(pairs, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def mix_sources(synthetic_pairs, real_pairs, real_weight: int = 4, seed: int = 0):
    combined = list(synthetic_pairs) + list(real_pairs) * real_weight
    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined
