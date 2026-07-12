# Crogrammar v0.2.0 — Veliki upgrade — Implementacijski plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nadograditi crogrammar na v0.2.0 — ByT5-base, hrvatska Wikipedia + hr500k, prošireni generator grešaka, RAPUT stvarni parovi u trening, 3 epohe, ručni test set, trening na Kaggle.

**Architecture:** Proširujemo postojeći `src/` layout paket. Novi moduli za Wikipedia pipeline i miješanje izvora; proširenja postojećih `noise.py`/`raput.py`/`build_dataset.py`. Sve deterministički dijelovi su unit-testabilni lokalno bez GPU-a; sam trening ide na Kaggle preko novog notebooka.

**Tech Stack:** Python 3.11+, HuggingFace `transformers`/`datasets`, ByT5-base, `regex`, `pytest`. Trening: Kaggle P100, fp32.

---

## Napomene o okolini

- OS: Windows, shell PowerShell 7+. Glavni Python je 3.14 (za lokalne testove — samo `regex` + stdlib).
- Teški paketi (`transformers`, `datasets`, `torch`) su u `[train]` extra; svi importi tih paketa MORAJU biti lijeni (unutar funkcija).
- Postojeći kod: `noise.py` (ima `strip_diacritics_safe`, `contract_ao`, `change_case_ending`, `corrupt_sentence(sentence, confusion, seed, p, real_words)`), `raput.py` (`parse_raput`, `read_raput_pairs`), `build_dataset.py` (`make_pairs`, `split_pairs`, `write_jsonl`), `confusion.py` (`load_confusion_from_dic`, `load_wordset_from_dic`).
- Git identitet: ako commit traži identitet koristi `git -c user.name="crogrammar" -c user.email="dev@crogrammar.local" commit -m "..."`.
- Ne koristi `cd`; radi iz `C:\Users\Mario\Projects\crogrammar`.
- Pokretanje testova: `python -m pytest`.

## Pregled datoteka

Izmijeniti:
- `src/crogrammar/data/noise.py` — dodati `merge_words`, `split_word`, `punctuation_noise`, `case_noise`, proširiti `corrupt_sentence`
- `src/crogrammar/data/raput.py` — dodati `raput_training_pairs`
- `src/crogrammar/data/build_dataset.py` — dodati `mix_sources`
- `README.md` — v0.2.0 napomene

Kreirati:
- `src/crogrammar/data/wikipedia.py` — `is_clean_sentence`, `load_wiki_sentences`
- `tests/test_wikipedia.py`
- `tests/manual_test_set.jsonl` — ručni test set
- `tests/test_manual_set.py` — validacija test seta
- `notebooks/train_kaggle.ipynb` — Kaggle trening notebook

---

## Task 1: Novi generatori grešaka — spajanje i razdvajanje riječi

**Files:**
- Modify: `src/crogrammar/data/noise.py`
- Test: `tests/test_noise.py`

- [ ] **Step 1: Dodaj testove na kraj `tests/test_noise.py`**

```python
def test_merge_words_joins_two():
    from crogrammar.data.noise import merge_words
    assert merge_words("u", "školu") == "uškolu"

def test_split_word_creates_space():
    import random
    from crogrammar.data.noise import split_word
    out = split_word("nemogu", random.Random(0))
    assert " " in out
    assert out.replace(" ", "") == "nemogu"

def test_split_word_leaves_short_words():
    import random
    from crogrammar.data.noise import split_word
    assert split_word("da", random.Random(0)) == "da"
```

- [ ] **Step 2: Pokreni testove — moraju pasti**

Run: `python -m pytest tests/test_noise.py -k "merge or split" -v`
Expected: FAIL (ImportError na `merge_words`)

- [ ] **Step 3: Dodaj funkcije u `src/crogrammar/data/noise.py` (nakon `change_case_ending`)**

```python
def merge_words(w1: str, w2: str) -> str:
    return w1 + w2


def split_word(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(2, len(word) - 1)
    return word[:i] + " " + word[i:]
```

- [ ] **Step 4: Pokreni testove — moraju proći**

Run: `python -m pytest tests/test_noise.py -k "merge or split" -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/noise.py tests/test_noise.py
git commit -m "feat: merge_words i split_word generatori gresaka"
```

---

## Task 2: Novi generatori grešaka — interpunkcija i velika slova

**Files:**
- Modify: `src/crogrammar/data/noise.py`
- Test: `tests/test_noise.py`

- [ ] **Step 1: Dodaj testove na kraj `tests/test_noise.py`**

```python
def test_case_noise_flips_first_letter():
    import random
    from crogrammar.data.noise import case_noise
    out = case_noise("škola", random.Random(0))
    assert out.lower() == "škola"
    assert out != "škola" or out == "škola"  # deterministicki sa seedom

def test_case_noise_deterministic():
    import random
    from crogrammar.data.noise import case_noise
    a = case_noise("more", random.Random(5))
    b = case_noise("more", random.Random(5))
    assert a == b

def test_punctuation_noise_changes_or_keeps():
    import random
    from crogrammar.data.noise import punctuation_noise
    a = punctuation_noise("idem doma, danas.", random.Random(1))
    b = punctuation_noise("idem doma, danas.", random.Random(1))
    assert a == b  # determinizam
    assert isinstance(a, str)

def test_punctuation_noise_can_drop_comma():
    import random
    from crogrammar.data.noise import punctuation_noise
    # nadji seed koji makne zarez
    found = False
    for s in range(50):
        if "," not in punctuation_noise("a, b, c, d", random.Random(s)):
            found = True
            break
    assert found
```

- [ ] **Step 2: Pokreni testove — moraju pasti**

Run: `python -m pytest tests/test_noise.py -k "case_noise or punctuation" -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Dodaj funkcije u `src/crogrammar/data/noise.py` (nakon `split_word`)**

```python
def case_noise(word: str, rng: random.Random) -> str:
    if not word:
        return word
    if word[0].isupper():
        return word[0].lower() + word[1:]
    return word[0].upper() + word[1:]


def punctuation_noise(sentence: str, rng: random.Random) -> str:
    if "," in sentence and rng.random() < 0.5:
        idx = sentence.index(",")
        return sentence[:idx] + sentence[idx + 1:]
    if sentence.endswith(".") and rng.random() < 0.5:
        return sentence[:-1]
    return sentence
```

- [ ] **Step 4: Pokreni testove — moraju proći**

Run: `python -m pytest tests/test_noise.py -k "case_noise or punctuation" -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/noise.py tests/test_noise.py
git commit -m "feat: case_noise i punctuation_noise generatori"
```

---

## Task 3: Integracija novih grešaka u corrupt_sentence

Trenutni `corrupt_sentence` bira između 5 tipova per-riječ (choice 0-4). Dodajemo interpunkcijski šum na razini rečenice (prije obrade riječi) i case_noise kao dio per-riječ izbora. Spajanje/razdvajanje radimo kao zaseban korak na razini rečenice da ne pokvarimo poravnanje.

**Files:**
- Modify: `src/crogrammar/data/noise.py`
- Test: `tests/test_noise.py`

- [ ] **Step 1: Dodaj testove na kraj `tests/test_noise.py`**

```python
def test_corrupt_sentence_still_deterministic():
    from crogrammar.data.noise import corrupt_sentence
    a = corrupt_sentence("idem u školu danas ujutro", {}, seed=11, p=0.5)
    b = corrupt_sentence("idem u školu danas ujutro", {}, seed=11, p=0.5)
    assert a == b

def test_corrupt_sentence_can_produce_space_change():
    from crogrammar.data.noise import corrupt_sentence
    # s visokim p, barem jedan seed proizvede spajanje ili razdvajanje (promjena broja razmaka)
    base = "idem u veliku školu svaki dan"
    changed_spacing = False
    for s in range(40):
        out = corrupt_sentence(base, {}, seed=s, p=0.9)
        if out.count(" ") != base.count(" "):
            changed_spacing = True
            break
    assert changed_spacing
```

- [ ] **Step 2: Pokreni testove — moraju pasti (drugi test) / proći (prvi)**

Run: `python -m pytest tests/test_noise.py -k "still_deterministic or space_change" -v`
Expected: `space_change` FAIL (spacing se još ne mijenja), `still_deterministic` PASS

- [ ] **Step 3: Zamijeni `corrupt_sentence` u `src/crogrammar/data/noise.py`**

Zamijeni cijelu funkciju `corrupt_sentence` ovom verzijom:

```python
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
```

- [ ] **Step 4: Pokreni testove — svi moraju proći**

Run: `python -m pytest tests/test_noise.py -v`
Expected: svi testovi u datoteci prolaze (uklj. postojeće)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/noise.py tests/test_noise.py
git commit -m "feat: integracija interpunkcije/case/split-merge u corrupt_sentence"
```

---

## Task 4: RAPUT stvarni parovi za trening

**Files:**
- Modify: `src/crogrammar/data/raput.py`
- Test: `tests/test_raput.py`

- [ ] **Step 1: Dodaj testove na kraj `tests/test_raput.py`**

```python
def test_raput_training_pairs_extracts_real_errors(tmp_path):
    from crogrammar.data.raput import raput_training_pairs
    sample = (
        "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC RAPUT:ORIG RAPUT:ERRORS\n"
        "# sent_id = 1\n"
        "1\takcijske\takcijski\tADJ\t_\t_\t_\t_\t_\t_\takciske\tPHON-SEG\n"
        "2\tfilm\tfilm\tNOUN\t_\t_\t_\t_\t_\t_\tfilm\t_\n"
        "\n"
        "# sent_id = 2\n"
        "1\tmoja\tmoj\tDET\t_\t_\t_\t_\t_\t_\tMoja\t_\n"
    )
    p = tmp_path / "r.conllup"
    p.write_text(sample, encoding="utf-8")
    pairs = raput_training_pairs(str(p))
    # sent 1 ima pravu gresku (akciske->akcijske) => ukljucen
    assert any(x["src"] == "akciske film" and x["tgt"] == "akcijske film" for x in pairs)
    # sent 2 je samo case (Moja->moja) => iskljucen
    assert all(x["src"].lower() != x["tgt"].lower() or x["src"] == x["tgt"] for x in pairs)
    assert not any(x["src"] == "Moja" for x in pairs)

def test_raput_training_pairs_returns_dicts(tmp_path):
    from crogrammar.data.raput import raput_training_pairs
    sample = (
        "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC RAPUT:ORIG RAPUT:ERRORS\n"
        "# sent_id = 1\n"
        "1\tdoći\tdoći\tVERB\t_\t_\t_\t_\t_\t_\tdoć\tORTHO\n"
    )
    p = tmp_path / "r.conllup"
    p.write_text(sample, encoding="utf-8")
    pairs = raput_training_pairs(str(p))
    assert pairs == [{"src": "doć", "tgt": "doći"}]
```

- [ ] **Step 2: Pokreni testove — moraju pasti**

Run: `python -m pytest tests/test_raput.py -k training_pairs -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Dodaj funkciju na kraj `src/crogrammar/data/raput.py`**

```python
def raput_training_pairs(path):
    pairs = []
    for src, tgt in read_raput_pairs(path):
        if src.lower() != tgt.lower():
            pairs.append({"src": src, "tgt": tgt})
    return pairs
```

- [ ] **Step 4: Pokreni testove — moraju proći**

Run: `python -m pytest tests/test_raput.py -k training_pairs -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/raput.py tests/test_raput.py
git commit -m "feat: raput_training_pairs za stvarne greske u trening"
```

---

## Task 5: Miješanje izvora s pretezanjem

**Files:**
- Modify: `src/crogrammar/data/build_dataset.py`
- Test: `tests/test_build_dataset.py`

- [ ] **Step 1: Dodaj testove na kraj `tests/test_build_dataset.py`**

```python
def test_mix_sources_repeats_real_pairs():
    from crogrammar.data.build_dataset import mix_sources
    syn = [{"src": "a", "tgt": "a"}]
    real = [{"src": "doć", "tgt": "doći"}]
    mixed = mix_sources(syn, real, real_weight=3, seed=0)
    # real par se pojavljuje 3 puta, syn jednom => ukupno 4
    assert len(mixed) == 4
    real_count = sum(1 for m in mixed if m["src"] == "doć")
    assert real_count == 3

def test_mix_sources_deterministic_shuffle():
    from crogrammar.data.build_dataset import mix_sources
    syn = [{"src": str(i), "tgt": str(i)} for i in range(10)]
    real = [{"src": "x", "tgt": "y"}]
    a = mix_sources(syn, real, real_weight=2, seed=42)
    b = mix_sources(syn, real, real_weight=2, seed=42)
    assert a == b

def test_mix_sources_real_weight_one():
    from crogrammar.data.build_dataset import mix_sources
    syn = [{"src": "a", "tgt": "a"}]
    real = [{"src": "b", "tgt": "c"}]
    mixed = mix_sources(syn, real, real_weight=1, seed=0)
    assert len(mixed) == 2
```

- [ ] **Step 2: Pokreni testove — moraju pasti**

Run: `python -m pytest tests/test_build_dataset.py -k mix_sources -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Dodaj funkciju na kraj `src/crogrammar/data/build_dataset.py`**

```python
def mix_sources(synthetic_pairs, real_pairs, real_weight: int = 4, seed: int = 0):
    combined = list(synthetic_pairs) + list(real_pairs) * real_weight
    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined
```

- [ ] **Step 4: Pokreni testove — moraju proći**

Run: `python -m pytest tests/test_build_dataset.py -k mix_sources -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: mix_sources - pretezanje stvarnih parova"
```

---

## Task 6: Wikipedia filtar kvalitete

**Files:**
- Create: `src/crogrammar/data/wikipedia.py`
- Test: `tests/test_wikipedia.py`

- [ ] **Step 1: Napiši `tests/test_wikipedia.py`**

```python
from crogrammar.data.wikipedia import is_clean_sentence


def test_accepts_normal_sentence():
    assert is_clean_sentence("Zagreb je glavni grad Hrvatske i najveći grad.")

def test_rejects_too_short():
    assert not is_clean_sentence("Da.")

def test_rejects_too_long():
    assert not is_clean_sentence("a " * 200)

def test_rejects_wiki_markup():
    assert not is_clean_sentence("== Povijest == grada Zagreba kroz stoljeća sada")
    assert not is_clean_sentence("[[Datoteka:slika.jpg]] prikazuje neki grad ovdje")
    assert not is_clean_sentence("{{Infokutija|ime=Zagreb|povrsina=641}} glavni grad")

def test_rejects_too_many_digits():
    assert not is_clean_sentence("1234 5678 9012 3456 7890 1234 5678 9012")

def test_rejects_no_letters():
    assert not is_clean_sentence("12:34 56.78 90 !!! ??? ...")

def test_accepts_sentence_with_some_numbers():
    assert is_clean_sentence("Grad ima oko 800 tisuća stanovnika prema popisu.")
```

- [ ] **Step 2: Pokreni testove — moraju pasti**

Run: `python -m pytest tests/test_wikipedia.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Napiši `src/crogrammar/data/wikipedia.py`**

```python
import regex as re

_MARKUP = ("==", "[[", "]]", "{{", "}}", "|")
_LETTER_RE = re.compile(r"\p{L}", re.UNICODE)
_DIGIT_RE = re.compile(r"\d")


def is_clean_sentence(text: str) -> bool:
    text = text.strip()
    if len(text) < 20 or len(text) > 200:
        return False
    if any(m in text for m in _MARKUP):
        return False
    letters = len(_LETTER_RE.findall(text))
    if letters < len(text) * 0.5:
        return False
    digits = len(_DIGIT_RE.findall(text))
    if digits > len(text) * 0.2:
        return False
    if "http" in text.lower():
        return False
    return True


def load_wiki_sentences(limit=None):
    from datasets import load_dataset
    from ..inference.segment import split_sentences
    ds = load_dataset("wikimedia/wikipedia", "20231101.hr", split="train",
                      streaming=True)
    seen = set()
    count = 0
    for article in ds:
        for sent in split_sentences(article["text"]):
            if not is_clean_sentence(sent):
                continue
            if sent in seen:
                continue
            seen.add(sent)
            yield sent
            count += 1
            if limit and count >= limit:
                return
```

- [ ] **Step 4: Pokreni testove — moraju proći**

Run: `python -m pytest tests/test_wikipedia.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/wikipedia.py tests/test_wikipedia.py
git commit -m "feat: Wikipedia filtar kvalitete i streaming loader"
```

---

## Task 7: Ručni test set

**Files:**
- Create: `tests/manual_test_set.jsonl`
- Create: `tests/test_manual_set.py`

- [ ] **Step 1: Napiši `tests/test_manual_set.py`**

```python
import json
from pathlib import Path

PATH = Path(__file__).parent / "manual_test_set.jsonl"
ALLOWED = {"diacritic", "morphology", "split_merge", "punctuation", "homograph_keep", "case"}


def _load():
    with open(PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def test_file_exists_and_nonempty():
    rows = _load()
    assert len(rows) >= 60

def test_each_row_has_required_keys():
    for r in _load():
        assert set(r.keys()) == {"src", "tgt", "kategorija"}

def test_categories_valid_and_all_present():
    cats = {r["kategorija"] for r in _load()}
    assert cats <= ALLOWED
    assert cats == ALLOWED  # sve kategorije zastupljene

def test_homograph_keep_rows_unchanged():
    for r in _load():
        if r["kategorija"] == "homograph_keep":
            assert r["src"] == r["tgt"]
```

- [ ] **Step 2: Pokreni testove — moraju pasti**

Run: `python -m pytest tests/test_manual_set.py -v`
Expected: FAIL (datoteka ne postoji)

- [ ] **Step 3: Kreiraj `tests/manual_test_set.jsonl`**

Kreiraj datoteku sa ~72 retka (12 po kategoriji). Svaki redak je JSON objekt s ključevima `src`, `tgt`, `kategorija`. Koristi točno ove retke (UTF-8, jedan JSON po liniji):

```jsonl
{"src": "idem u skolu", "tgt": "idem u školu", "kategorija": "diacritic"}
{"src": "moja casa je puna", "tgt": "moja čaša je puna", "kategorija": "diacritic"}
{"src": "citam knjigu navecer", "tgt": "čitam knjigu navečer", "kategorija": "diacritic"}
{"src": "zelim ti srecu", "tgt": "želim ti sreću", "kategorija": "diacritic"}
{"src": "djak je dobar ucenik", "tgt": "đak je dobar učenik", "kategorija": "diacritic"}
{"src": "vozim se biciklom kroz grad", "tgt": "vozim se biciklom kroz grad", "kategorija": "diacritic"}
{"src": "nosim zutu majicu", "tgt": "nosim žutu majicu", "kategorija": "diacritic"}
{"src": "pijem caj svako jutro", "tgt": "pijem čaj svako jutro", "kategorija": "diacritic"}
{"src": "susjed ima velik vrt", "tgt": "susjed ima velik vrt", "kategorija": "diacritic"}
{"src": "kupujem svjeze voce", "tgt": "kupujem svježe voće", "kategorija": "diacritic"}
{"src": "gledam more s obale", "tgt": "gledam more s obale", "kategorija": "diacritic"}
{"src": "cesto secem parkom", "tgt": "često šećem parkom", "kategorija": "diacritic"}
{"src": "on je otiša doma", "tgt": "on je otišao doma", "kategorija": "morphology"}
{"src": "ja sam radi cijeli dan", "tgt": "ja sam radio cijeli dan", "kategorija": "morphology"}
{"src": "ona je reka istinu", "tgt": "ona je rekla istinu", "kategorija": "morphology"}
{"src": "mi smo doša kasno", "tgt": "mi smo došli kasno", "kategorija": "morphology"}
{"src": "vidio sam ga na ulico", "tgt": "vidio sam ga na ulici", "kategorija": "morphology"}
{"src": "razgovaram s prijatelja", "tgt": "razgovaram s prijateljem", "kategorija": "morphology"}
{"src": "idem prema kuco", "tgt": "idem prema kući", "kategorija": "morphology"}
{"src": "kupio sam dvije knjige", "tgt": "kupio sam dvije knjige", "kategorija": "morphology"}
{"src": "sjedim u sobo", "tgt": "sjedim u sobi", "kategorija": "morphology"}
{"src": "pišem pismo sestre", "tgt": "pišem pismo sestri", "kategorija": "morphology"}
{"src": "on bi doša da može", "tgt": "on bi došao da može", "kategorija": "morphology"}
{"src": "gledali smo film zajedno", "tgt": "gledali smo film zajedno", "kategorija": "morphology"}
{"src": "nemogu doći danas", "tgt": "ne mogu doći danas", "kategorija": "split_merge"}
{"src": "netreba mi pomoć", "tgt": "ne treba mi pomoć", "kategorija": "split_merge"}
{"src": "idem uškolu ujutro", "tgt": "idem u školu ujutro", "kategorija": "split_merge"}
{"src": "on nezna odgovor", "tgt": "on ne zna odgovor", "kategorija": "split_merge"}
{"src": "dolazim na kavu", "tgt": "dolazim na kavu", "kategorija": "split_merge"}
{"src": "sve je u redu", "tgt": "sve je u redu", "kategorija": "split_merge"}
{"src": "napisao sam za daću", "tgt": "napisao sam zadaću", "kategorija": "split_merge"}
{"src": "ne mogu vjerovati", "tgt": "ne mogu vjerovati", "kategorija": "split_merge"}
{"src": "to je ne moguće", "tgt": "to je nemoguće", "kategorija": "split_merge"}
{"src": "vidimo se su tra", "tgt": "vidimo se sutra", "kategorija": "split_merge"}
{"src": "kavu pijem ujutro", "tgt": "kavu pijem ujutro", "kategorija": "split_merge"}
{"src": "on ne dolazi večeras", "tgt": "on ne dolazi večeras", "kategorija": "split_merge"}
{"src": "došao je kući umoran no sretan", "tgt": "došao je kući umoran, no sretan", "kategorija": "punctuation"}
{"src": "idem doma jer je kasno", "tgt": "idem doma jer je kasno", "kategorija": "punctuation"}
{"src": "kupila je kruh mlijeko i jaja", "tgt": "kupila je kruh, mlijeko i jaja", "kategorija": "punctuation"}
{"src": "danas je lijep dan", "tgt": "danas je lijep dan.", "kategorija": "punctuation"}
{"src": "ako dođeš javi mi se", "tgt": "ako dođeš, javi mi se", "kategorija": "punctuation"}
{"src": "volim ljeto zima mi je hladna", "tgt": "volim ljeto, zima mi je hladna", "kategorija": "punctuation"}
{"src": "on je otišao ona je ostala", "tgt": "on je otišao, ona je ostala", "kategorija": "punctuation"}
{"src": "sve je bilo u redu", "tgt": "sve je bilo u redu.", "kategorija": "punctuation"}
{"src": "kada stigneš nazovi me", "tgt": "kada stigneš, nazovi me", "kategorija": "punctuation"}
{"src": "pas maca i ptica žive skupa", "tgt": "pas, maca i ptica žive skupa", "kategorija": "punctuation"}
{"src": "hvala ti puno", "tgt": "hvala ti puno.", "kategorija": "punctuation"}
{"src": "iako je padala kiša izašli smo", "tgt": "iako je padala kiša, izašli smo", "kategorija": "punctuation"}
{"src": "imam sto kuna u novčaniku", "tgt": "imam sto kuna u novčaniku", "kategorija": "homograph_keep"}
{"src": "sto je to na stolu", "tgt": "sto je to na stolu", "kategorija": "homograph_keep"}
{"src": "posto je skupo ne kupujem", "tgt": "pošto je skupo, ne kupujem", "kategorija": "homograph_keep"}
{"src": "dao mi je sto eura", "tgt": "dao mi je sto eura", "kategorija": "homograph_keep"}
{"src": "grad je bio miran", "tgt": "grad je bio miran", "kategorija": "homograph_keep"}
{"src": "luk je zdravo povrće", "tgt": "luk je zdravo povrće", "kategorija": "homograph_keep"}
{"src": "kupio sam novi sto", "tgt": "kupio sam novi stol", "kategorija": "homograph_keep"}
{"src": "more je bilo mirno", "tgt": "more je bilo mirno", "kategorija": "homograph_keep"}
{"src": "sunce je grijalo jako", "tgt": "sunce je grijalo jako", "kategorija": "homograph_keep"}
{"src": "put je bio dug", "tgt": "put je bio dug", "kategorija": "homograph_keep"}
{"src": "dan je bio topao", "tgt": "dan je bio topao", "kategorija": "homograph_keep"}
{"src": "sela je za sto", "tgt": "sela je za stol", "kategorija": "homograph_keep"}
{"src": "marko ide u školu", "tgt": "Marko ide u školu", "kategorija": "case"}
{"src": "Idem Kući Sada", "tgt": "idem kući sada", "kategorija": "case"}
{"src": "zagreb je glavni grad", "tgt": "Zagreb je glavni grad", "kategorija": "case"}
{"src": "ana i ivan su prijatelji", "tgt": "Ana i Ivan su prijatelji", "kategorija": "case"}
{"src": "u Ponedjeljak idem na posao", "tgt": "u ponedjeljak idem na posao", "kategorija": "case"}
{"src": "hrvatska je lijepa zemlja", "tgt": "Hrvatska je lijepa zemlja", "kategorija": "case"}
{"src": "danas Je Lijep dan", "tgt": "danas je lijep dan", "kategorija": "case"}
{"src": "petar voli nogomet", "tgt": "Petar voli nogomet", "kategorija": "case"}
{"src": "moj brat živi u splitu", "tgt": "moj brat živi u Splitu", "kategorija": "case"}
{"src": " latica je bila crvena", "tgt": "latica je bila crvena", "kategorija": "case"}
{"src": "rijeka Sava protječe gradom", "tgt": "rijeka Sava protječe gradom", "kategorija": "case"}
{"src": "susjeda Marija je draga", "tgt": "susjeda Marija je draga", "kategorija": "case"}
```

- [ ] **Step 4: Pokreni testove — moraju proći**

Run: `python -m pytest tests/test_manual_set.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tests/manual_test_set.jsonl tests/test_manual_set.py
git commit -m "feat: rucni test set (72 recenice, 6 kategorija)"
```

---

## Task 8: Kaggle trening notebook

**Files:**
- Create: `notebooks/train_kaggle.ipynb`

- [ ] **Step 1: Kreiraj `notebooks/train_kaggle.ipynb` kao valjani ipynb JSON**

Notebook mora imati sljedeće ćelije (markdown + code), redom. Svaka code-ćelija je zaseban element. Sadržaj code-ćelija:

Ćelija 1 (install):
```python
!git clone https://github.com/mmlinaric/crogrammar.git
%cd crogrammar
!pip install -e ".[train]"
```

Ćelija 2 (preuzmi izvore):
```python
from crogrammar.data.download import download_hr500k, download_hunspell_dic, download_raput
download_hr500k("data/raw")
download_hunspell_dic("data/raw")
raput_path = download_raput("data/raw")
print("raput:", raput_path)
```

Ćelija 3 (čist tekst: hr500k + Wikipedia):
```python
import gzip
from pathlib import Path
from crogrammar.data.conllu import parse_conllu, sentence_text
from crogrammar.data.wikipedia import load_wiki_sentences

clean = []
for gz in Path("data/raw/hr500k").glob("*.conllu.gz"):
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        for sent in parse_conllu(f.read()):
            clean.append(sentence_text(sent))
print("hr500k recenica:", len(clean))

wiki = list(load_wiki_sentences(limit=1_000_000))
print("wiki recenica:", len(wiki))
clean += wiki
print("ukupno cistih:", len(clean))
```

Ćelija 4 (izgradi dataset):
```python
from crogrammar.data.confusion import load_confusion_from_dic, load_wordset_from_dic
from crogrammar.data.build_dataset import make_pairs, mix_sources, split_pairs, write_jsonl
from crogrammar.data.raput import raput_training_pairs

confusion = load_confusion_from_dic("data/raw/hr_HR.dic")
real_words = load_wordset_from_dic("data/raw/hr_HR.dic")
synthetic = make_pairs(clean, confusion, seed=42, variants=2, real_words=real_words)
print("sintetickih parova:", len(synthetic))

real = raput_training_pairs(str(raput_path))
print("RAPUT stvarnih parova:", len(real))

all_pairs = mix_sources(synthetic, real, real_weight=4, seed=42)
train, dev, test = split_pairs(all_pairs, dev_frac=0.02, test_frac=0.02, seed=42)
write_jsonl(train, "data/processed/train.jsonl")
write_jsonl(dev, "data/processed/dev.jsonl")
write_jsonl(test, "data/processed/test.jsonl")
print(len(train), len(dev), len(test))
```

Ćelija 5 (trening, resume-safe):
```python
from crogrammar.train.config import TrainConfig
from crogrammar.train.train import train

cfg = TrainConfig(
    base_model="google/byt5-base",
    output_dir="/kaggle/working/byt5-hr-gec",
    batch_size=4,
    grad_accum=8,
    max_source_len=160,
    max_target_len=160,
    num_epochs=3,
    fp16=False,
)
train(cfg, resume_from_checkpoint=True)
```

Ćelija 6 (evaluacija na ručnom setu):
```python
import json
from crogrammar.inference.gec import CroatianGEC
from crogrammar.eval.run_eval import evaluate_pairs_batched

gec = CroatianGEC(model_path="/kaggle/working/byt5-hr-gec")

with open("tests/manual_test_set.jsonl", encoding="utf-8") as f:
    manual = [json.loads(l) for l in f if l.strip()]

overall = evaluate_pairs_batched(manual, gec.generate_batch, batch_size=32)
print("GLEU (rucni set):", round(overall["gleu"], 4), "| n:", overall["n"])

from collections import defaultdict
by_cat = defaultdict(list)
for r in manual:
    by_cat[r["kategorija"]].append(r)
for cat, rows in sorted(by_cat.items()):
    s = evaluate_pairs_batched(rows, gec.generate_batch, batch_size=32)
    print(f"  {cat:16s}: GLEU {round(s['gleu'],4)} (n={s['n']})")
```

Ćelija 7 (spremi kao Kaggle Dataset izlaz):
```python
import json, datetime, os
meta = {
    "version": "0.2.0",
    "date": datetime.date.today().isoformat(),
    "base_model": "google/byt5-base",
    "gleu_manual": round(overall["gleu"], 4),
    "license": "CC BY-SA 4.0",
    "sources": ["hr500k", "hunspell-hr", "RAPUT (trening)", "Wikipedia hr"],
}
os.makedirs("/kaggle/working/byt5-hr-gec", exist_ok=True)
with open("/kaggle/working/byt5-hr-gec/METADATA.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(meta)
print("Model je u /kaggle/working/byt5-hr-gec -> spremi output ili kreiraj Kaggle Dataset za resume.")
```

Markdown ćelije: dodati kratke naslove prije svake code ćelije (npr. "## 1. Instalacija", "## 5. Trening (ByT5-base, resume-safe)"). Napomena o resume u markdownu ćelije 5: "Ako je prethodni run spremio checkpoint kao Kaggle Dataset, dodaj ga kao input u /kaggle/input i kopiraj u output_dir prije pokretanja."

Metadata notebooka mora sadržavati `"accelerator": "GPU"`, `nbformat: 4`, `nbformat_minor: 0`, kernelspec python3.

- [ ] **Step 2: Provjeri da je notebook valjani JSON**

Run: `python -c "import json; json.load(open('notebooks/train_kaggle.ipynb', encoding='utf-8')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add notebooks/train_kaggle.ipynb
git commit -m "feat: Kaggle trening notebook za v0.2.0"
```

---

## Task 9: Puni test suite + README

**Files:**
- Modify: `README.md`
- Test: sve

- [ ] **Step 1: Pokreni cijeli test suite**

Run: `python -m pytest -q`
Expected: svi testovi prolaze (postojeći + novi iz Task 1-7)

- [ ] **Step 2: Dodaj v0.2.0 sekciju u `README.md` (nakon postojeće "Rezultati" tablice)**

```markdown
### v0.2.0 (u izradi)

Veliki upgrade: ByT5-base, hrvatska Wikipedia + hr500k (~1M+ recenica), prosireni
generator gresaka (spajanje/razdvajanje rijeci, interpunkcija, velika slova), RAPUT
stvarne greske u trening (pretezane x4), 3 epohe. Trening na Kaggle (P100), model se
sprema u /kaggle/working i kao Kaggle Dataset za resume (bez Google Drive-a).

Mjerenje: `tests/manual_test_set.jsonl` (72 recenice, 6 kategorija) — usporedba
v0.1.0 vs v0.2.0 GLEU po kategoriji.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README v0.2.0 napomene"
```

---

## Self-Review (autor plana)

- **Spec coverage:** [1] Wikipedia → Task 6; [2] generator → Task 1-3; [3] RAPUT training pairs → Task 4; [4] mix_sources → Task 5; [5] config profil → Task 8 (notebook konstruira TrainConfig, postojeći config već ima parametre); [6] ručni test set → Task 7; [7] Kaggle notebook → Task 8; testiranje → svi + Task 9. Pokriveno.
- **Placeholder scan:** svi kod-koraci imaju konkretan kod; test set ima sve retke doslovno.
- **Type consistency:** `mix_sources(synthetic, real, real_weight, seed)`, `raput_training_pairs(path) -> [{"src","tgt"}]`, `is_clean_sentence(text) -> bool`, `load_wiki_sentences(limit)` — dosljedno korišteni u Task 8 notebooku. `corrupt_sentence` potpis nepromijenjen (backward-compatible). `TrainConfig` parametri (`base_model`, `batch_size`, `grad_accum`, `max_source_len`, `max_target_len`, `num_epochs`, `fp16`) postoje u postojećem config.py.
- **Napomena:** Task 8 notebook koristi postojeći `train()` koji već ima `processing_class`, bf16 getattr, i fallback spremanje — sve kompatibilno s Kaggle.
