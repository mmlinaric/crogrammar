import json

from .metrics import gleu_sentence, gleu_sentence_diacritic_insensitive


def _score_fn(diacritic_insensitive):
    return gleu_sentence_diacritic_insensitive if diacritic_insensitive else gleu_sentence


def evaluate_pairs(pairs, generate_fn, diacritic_insensitive=False):
    score = _score_fn(diacritic_insensitive)
    scores = []
    for p in pairs:
        hyp = generate_fn(p["src"])
        scores.append(score(hyp, p["tgt"]))
    n = len(scores)
    return {"gleu": sum(scores) / n if n else 0.0, "n": n}


def evaluate_pairs_batched(pairs, generate_batch, batch_size=32, progress=False,
                           diacritic_insensitive=False):
    score = _score_fn(diacritic_insensitive)
    scores = []
    starts = range(0, len(pairs), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm
            starts = tqdm(starts, total=(len(pairs) + batch_size - 1) // batch_size)
        except ImportError:
            pass
    for i in starts:
        chunk = pairs[i:i + batch_size]
        hyps = generate_batch([p["src"] for p in chunk])
        for hyp, p in zip(hyps, chunk):
            scores.append(score(hyp, p["tgt"]))
    n = len(scores)
    return {"gleu": sum(scores) / n if n else 0.0, "n": n}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def raput_to_pairs(conllup_path):
    from ..data.raput import read_raput_pairs
    return [{"src": s, "tgt": t} for s, t in read_raput_pairs(conllup_path)]
