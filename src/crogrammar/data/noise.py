import random

_DIA = str.maketrans({"š": "s", "Š": "S", "ž": "z", "Ž": "Z",
                      "č": "c", "Č": "C", "ć": "c", "Ć": "C"})


def strip_diacritics(text: str) -> str:
    text = text.translate(_DIA)
    return text.replace("đ", "dj").replace("Đ", "Dj")


def strip_diacritics_safe(word: str, real_words) -> str:
    stripped = strip_diacritics(word)
    if stripped == word:
        return word
    if real_words and stripped in real_words:
        return word
    return stripped


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


def merge_words(w1: str, w2: str) -> str:
    return w1 + w2


def split_word(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(2, len(word) - 1)
    return word[:i] + " " + word[i:]


def case_noise(word: str, rng: random.Random) -> str:
    if not word:
        return word
    if word[0].isupper():
        return word[0].lower() + word[1:]
    return word[0].upper() + word[1:]


def punctuation_noise(sentence: str, rng: random.Random) -> str:
    if "," in sentence and rng.random() < 0.5:
        return sentence.replace(",", "")
    if sentence.endswith(".") and rng.random() < 0.5:
        return sentence[:-1]
    return sentence


def corrupt_sentence(sentence: str, confusion: dict, seed: int, p: float = 0.3,
                     real_words=None) -> str:
    rng = random.Random(seed)
    sentence = punctuation_noise(sentence, rng)
    words = sentence.split()
    out = []
    changed = False
    for w in words:
        r = rng.random()
        if r < p:
            choice = rng.randrange(6)
            if choice == 0:
                nw = apply_confusion(w, confusion, rng)
                if nw == w:
                    nw = strip_diacritics_safe(w, real_words)
            elif choice == 1:
                nw = strip_diacritics_safe(w, real_words)
            elif choice == 2:
                nw = typo_swap(w, rng)
            elif choice == 3:
                nw = contract_ao(w)
                if nw == w:
                    nw = strip_diacritics_safe(w, real_words)
            elif choice == 4:
                nw = change_case_ending(w, rng)
                if nw == w:
                    nw = strip_diacritics_safe(w, real_words)
            else:
                nw = case_noise(w, rng)
            if nw != w:
                changed = True
            out.append(nw)
        else:
            out.append(w)
    # spajanje/razdvajanje na razini rečenice
    if len(out) >= 2 and rng.random() < p * 0.5:
        i = rng.randrange(len(out) - 1)
        out[i] = merge_words(out[i], out[i + 1])
        del out[i + 1]
        changed = True
    elif out and rng.random() < p * 0.5:
        i = rng.randrange(len(out))
        split = split_word(out[i], rng)
        if split != out[i]:
            out[i] = split
            changed = True
    if not changed and words:
        out[0] = strip_diacritics_safe(out[0], real_words) or out[0]
    return " ".join(out)
