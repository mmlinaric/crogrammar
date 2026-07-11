import json

from .metrics import gleu_sentence


def evaluate_pairs(pairs, generate_fn):
    scores = []
    for p in pairs:
        hyp = generate_fn(p["src"])
        scores.append(gleu_sentence(hyp, p["tgt"]))
    n = len(scores)
    return {"gleu": sum(scores) / n if n else 0.0, "n": n}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
