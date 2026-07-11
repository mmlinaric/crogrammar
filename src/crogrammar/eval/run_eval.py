import json

from .metrics import gleu_sentence


def evaluate_pairs(pairs, generate_fn):
    scores = []
    for p in pairs:
        hyp = generate_fn(p["src"])
        scores.append(gleu_sentence(hyp, p["tgt"]))
    n = len(scores)
    return {"gleu": sum(scores) / n if n else 0.0, "n": n}


def evaluate_pairs_batched(pairs, generate_batch, batch_size=32):
    scores = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        hyps = generate_batch([p["src"] for p in chunk])
        for hyp, p in zip(hyps, chunk):
            scores.append(gleu_sentence(hyp, p["tgt"]))
    n = len(scores)
    return {"gleu": sum(scores) / n if n else 0.0, "n": n}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
