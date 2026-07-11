import random

_DIA = str.maketrans({"š": "s", "Š": "S", "ž": "z", "Ž": "Z",
                      "č": "c", "Č": "C", "ć": "c", "Ć": "C"})


def strip_diacritics(text: str) -> str:
    text = text.translate(_DIA)
    return text.replace("đ", "dj").replace("Đ", "Dj")


def typo_swap(word: str, rng: random.Random) -> str:
    if len(word) < 2:
        return word
    i = rng.randrange(len(word) - 1)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def apply_confusion(word: str, confusion: dict, rng: random.Random) -> str:
    errs = confusion.get(word)
    if not errs:
        return word
    return rng.choice(errs[: min(3, len(errs))])


def corrupt_sentence(sentence: str, confusion: dict, seed: int, p: float = 0.3) -> str:
    rng = random.Random(seed)
    words = sentence.split()
    out = []
    changed = False
    for w in words:
        r = rng.random()
        if r < p:
            choice = rng.randrange(3)
            if choice == 0:
                nw = apply_confusion(w, confusion, rng)
                if nw == w:
                    nw = strip_diacritics(w)
            elif choice == 1:
                nw = strip_diacritics(w)
            else:
                nw = typo_swap(w, rng)
            if nw != w:
                changed = True
            out.append(nw)
        else:
            out.append(w)
    if not changed and words:
        out[0] = strip_diacritics(out[0]) or out[0]
    return " ".join(out)
