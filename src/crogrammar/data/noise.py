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


_VOWELS = "aeiouAEIOU"


def contract_ao(word: str) -> str:
    if len(word) >= 3 and word[-2:] in ("ao", "io"):
        return word[:-1]
    return word


def change_case_ending(word: str, rng: random.Random) -> str:
    if len(word) < 2 or word[-1] not in _VOWELS:
        return word
    options = [v for v in "aeiou" if v != word[-1].lower()]
    return word[:-1] + rng.choice(options)


def corrupt_sentence(sentence: str, confusion: dict, seed: int, p: float = 0.3) -> str:
    rng = random.Random(seed)
    words = sentence.split()
    out = []
    changed = False
    for w in words:
        r = rng.random()
        if r < p:
            choice = rng.randrange(5)
            if choice == 0:
                nw = apply_confusion(w, confusion, rng)
                if nw == w:
                    nw = strip_diacritics(w)
            elif choice == 1:
                nw = strip_diacritics(w)
            elif choice == 2:
                nw = typo_swap(w, rng)
            elif choice == 3:
                nw = contract_ao(w)
                if nw == w:
                    nw = strip_diacritics(w)
            else:
                nw = change_case_ending(w, rng)
                if nw == w:
                    nw = strip_diacritics(w)
            if nw != w:
                changed = True
            out.append(nw)
        else:
            out.append(w)
    if not changed and words:
        out[0] = strip_diacritics(out[0]) or out[0]
    return " ".join(out)
