from collections import Counter

from ..data.noise import strip_diacritics


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def gleu_sentence(hypothesis: str, reference: str, max_n: int = 4) -> float:
    hyp = hypothesis.split()
    ref = reference.split()
    if not hyp or not ref:
        return 0.0
    total_overlap = 0
    total_hyp = 0
    for n in range(1, max_n + 1):
        h = _ngrams(hyp, n)
        r = _ngrams(ref, n)
        overlap = sum((h & r).values())
        total_overlap += overlap
        total_hyp += max(sum(h.values()), 0)
    if total_hyp == 0:
        return 0.0
    return total_overlap / total_hyp


def gleu_sentence_diacritic_insensitive(hypothesis: str, reference: str, max_n: int = 4) -> float:
    return gleu_sentence(strip_diacritics(hypothesis.lower()),
                         strip_diacritics(reference.lower()), max_n=max_n)


def prf_edits(pred: set, gold: set, beta: float = 0.5):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    if p == 0 and r == 0:
        return p, r, 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0
    return p, r, f
