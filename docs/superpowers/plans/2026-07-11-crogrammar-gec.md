# Crogrammar — Hrvatski GEC — Implementacijski plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Izgraditi prenosiv neuronski GEC model za hrvatski (ByT5 fine-tuned) s podatkovnim pipelineom, treningom, evaluacijom i čistim inference sučeljem `correct(text) -> Result`.

**Architecture:** Četiri neovisne cjeline — [1] podatkovni pipeline (sirovi korpusi → parovi grešaka), [2] trening (fine-tune ByT5 na Colab/Kaggle), [3] model kao HuggingFace artefakt, [4] inference sučelje (segmentacija + model + diff → izmjene). Cjeline 1, 3, 4, eval razvijaju se i testiraju bez GPU-a; GPU treba tek za trening.

**Tech Stack:** Python 3.11+, HuggingFace `transformers` + `datasets`, `classla` (hrvatski NLP), `sacrebleu` (GLEU), `pytest`, ByT5-small.

---

## Napomene o okolini

- OS: Windows, shell: PowerShell 7+. Naredbe u planu su cross-platform (pytest, git, python).
- Paket koristi `src/` layout; instalira se editabilno (`pip install -e .`).
- Sve datoteke koriste UTF-8. Hrvatski tekst (č/ć/đ/š/ž) mora se čuvati bez ASCII normalizacije.
- Veliki podaci (`data/raw/`, `data/processed/`, `models/`) su u `.gitignore`.

## Pregled datoteka

Kreirati:
- `pyproject.toml` — definicija paketa i ovisnosti
- `.gitignore`
- `README.md`
- `src/crogrammar/__init__.py`
- `src/crogrammar/data/__init__.py`
- `src/crogrammar/data/conllu.py` — parsiranje CoNLL-U (hr500k)
- `src/crogrammar/data/confusion.py` — učitavanje ispravi.me confusion seta
- `src/crogrammar/data/noise.py` — generatori sintetičkih grešaka
- `src/crogrammar/data/build_dataset.py` — orkestracija → train/dev/test.jsonl
- `src/crogrammar/data/download.py` — preuzimanje sirovih izvora
- `src/crogrammar/inference/__init__.py`
- `src/crogrammar/inference/segment.py` — segmentacija na rečenice
- `src/crogrammar/inference/diff.py` — token-level diff → edits + tipizacija
- `src/crogrammar/inference/gec.py` — klasa `CroatianGEC` i `Result`
- `src/crogrammar/train/__init__.py`
- `src/crogrammar/train/config.py` — dataclass s hiperparametrima
- `src/crogrammar/train/train.py` — trening petlja (Seq2SeqTrainer)
- `src/crogrammar/eval/__init__.py`
- `src/crogrammar/eval/metrics.py` — GLEU + P/R/F0.5
- `notebooks/train_colab.ipynb` — Colab/Kaggle notebook (opisan, ne izvršava se u testovima)
- `tests/` — po jedan test modul za svaku determinističku komponentu

---

## Faza 0 — Skele projekta

### Task 1: Inicijalizacija paketa i ovisnosti

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/crogrammar/__init__.py`

- [ ] **Step 1: Napiši `pyproject.toml`**

```toml
[project]
name = "crogrammar"
version = "0.1.0"
description = "Hrvatski gramaticki checker (GEC) - neuronski seq2seq model"
requires-python = ">=3.11"
dependencies = [
    "regex>=2024.5",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
train = [
    "transformers>=4.44",
    "datasets>=2.20",
    "torch>=2.2",
    "sentencepiece>=0.2",
    "sacrebleu>=2.4",
    "classla>=2.1",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

> **Napomena:** Teški ML paketi (`transformers`, `torch`, `datasets`, `classla`,
> `sentencepiece`) su u opcionalnom `[train]` extra jer trebaju samo za trening
> (Colab/Kaggle). Jezgra i lokalni testovi (Faze 1–3, eval) rade samo s `regex`
> i standardnom knjižnicom, pa se sve razvija i testira lokalno bez GPU-a i bez
> teških ovisnosti. Import `transformers`/`classla` u kodu mora biti **lijen**
> (unutar funkcije), a ne na vrhu modula.

- [ ] **Step 2: Napiši `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
venv/
data/raw/
data/processed/
models/
*.gz
*.jsonl
```

- [ ] **Step 3: Napiši `src/crogrammar/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Instaliraj paket editabilno i provjeri**

Run: `pip install -e ".[dev]"`
Expected: uspješna instalacija, `crogrammar` importabilan.

- [ ] **Step 5: Provjeri import**

Run: `python -c "import crogrammar; print(crogrammar.__version__)"`
Expected: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/crogrammar/__init__.py
git commit -m "chore: skele projekta i ovisnosti"
```

---

## Faza 1 — Podatkovni pipeline

### Task 2: Parsiranje CoNLL-U (hr500k)

hr500k dolazi u CoNLL-U formatu: rečenice odvojene praznim redom, svaki token je redak
s tab-odvojenim poljima (ID, FORM, LEMMA, UPOS, XPOS, FEATS, ...). Komentari počinju s `#`.
Trebamo izvući čiste rečenice (rekonstruirane iz FORM tokena) i morfološke oznake.

**Files:**
- Create: `src/crogrammar/data/__init__.py` (prazan)
- Create: `src/crogrammar/data/conllu.py`
- Test: `tests/test_conllu.py`

- [ ] **Step 1: Napiši failing test**

```python
from crogrammar.data.conllu import parse_conllu, Token

SAMPLE = """# sent_id = 1
# text = On ide.
1\tOn\ton\tPRON\tPp3msn\tCase=Nom|Gender=Masc\t_\t_
2\tide\tici\tVERB\tVmr3s\tPerson=3|Number=Sing\t_\t_
3\t.\t.\tPUNCT\tZ\t_\t_\t_

# sent_id = 2
# text = Ja spavam.
1\tJa\tja\tPRON\tPp1-sn\tCase=Nom\t_\t_
2\tspavam\tspavati\tVERB\tVmr1s\tPerson=1|Number=Sing\t_\t_
3\t.\t.\tPUNCT\tZ\t_\t_\t_
"""

def test_parse_conllu_returns_sentences():
    sentences = list(parse_conllu(SAMPLE))
    assert len(sentences) == 2

def test_sentence_has_tokens_with_features():
    sentences = list(parse_conllu(SAMPLE))
    first = sentences[0]
    assert first[0] == Token(form="On", lemma="on", upos="PRON", feats={"Case": "Nom", "Gender": "Masc"})
    assert first[1].form == "ide"

def test_reconstruct_text_joins_forms():
    from crogrammar.data.conllu import sentence_text
    sentences = list(parse_conllu(SAMPLE))
    assert sentence_text(sentences[0]) == "On ide ."
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_conllu.py -v`
Expected: FAIL (`ModuleNotFoundError` / `ImportError`)

- [ ] **Step 3: Implementiraj `conllu.py`**

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Token:
    form: str
    lemma: str
    upos: str
    feats: dict = field(default_factory=dict)


def _parse_feats(raw: str) -> dict:
    if raw == "_" or not raw:
        return {}
    out = {}
    for pair in raw.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
    return out


def parse_conllu(text: str):
    tokens = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if not line.strip():
            if tokens:
                yield tokens
                tokens = []
            continue
        cols = line.split("\t")
        if len(cols) < 6:
            continue
        if "-" in cols[0] or "." in cols[0]:
            continue
        tokens.append(Token(form=cols[1], lemma=cols[2], upos=cols[3], feats=_parse_feats(cols[5])))
    if tokens:
        yield tokens


def sentence_text(tokens) -> str:
    return " ".join(t.form for t in tokens)
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_conllu.py -v`
Expected: PASS (3 testa)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/__init__.py src/crogrammar/data/conllu.py tests/test_conllu.py
git commit -m "feat: parsiranje CoNLL-U (hr500k)"
```

---

### Task 3: Confusion set iz ispravi.me

ispravi.me dataset: gzip TSV, stupci `datum\tgreska\tispravak`. Gradimo mapiranje
`ispravak -> [česte greške]` i obrnuto, za korištenje u generiranju sintetičkih grešaka.
Filtriramo na parove edit-distance male, i pamtimo frekvenciju.

**Files:**
- Create: `src/crogrammar/data/confusion.py`
- Test: `tests/test_confusion.py`

- [ ] **Step 1: Napiši failing test**

```python
from crogrammar.data.confusion import build_confusion_set

ROWS = [
    ("2020-01-01", "skola", "škola"),
    ("2020-01-02", "skola", "škola"),
    ("2020-01-03", "otisao", "otišao"),
    ("2020-01-04", "škola", "škola"),  # ista rijec, mora se preskociti
]

def test_maps_correct_to_errors():
    cs = build_confusion_set(ROWS)
    assert "škola" in cs
    assert "skola" in cs["škola"]

def test_skips_identical_pairs():
    cs = build_confusion_set(ROWS)
    assert "škola" not in cs.get("škola", [])

def test_orders_errors_by_frequency():
    cs = build_confusion_set(ROWS)
    assert cs["škola"][0] == "skola"  # 2 pojave, na prvom mjestu

def test_min_frequency_filter():
    cs = build_confusion_set(ROWS, min_freq=2)
    assert "škola" in cs        # skola ima 2 pojave
    assert "otišao" not in cs   # otisao ima 1 pojavu
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_confusion.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `confusion.py`**

```python
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
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_confusion.py -v`
Expected: PASS (4 testa)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/confusion.py tests/test_confusion.py
git commit -m "feat: confusion set iz ispravi.me dataseta"
```

---

### Task 4: Generatori sintetičkih grešaka

Determinističke funkcije koje primaju čistu rečenicu (+ opcionalno confusion set) i vraćaju
pogrešnu verziju. Koristimo seedani RNG za reproducibilnost. Četiri tipa: dijakritika,
tipografske, confusion-set zamjene, morfološke (jednostavna varijanta: brisanje sufiksa).

**Files:**
- Create: `src/crogrammar/data/noise.py`
- Test: `tests/test_noise.py`

- [ ] **Step 1: Napiši failing test**

```python
import random
from crogrammar.data.noise import (
    strip_diacritics,
    typo_swap,
    apply_confusion,
    corrupt_sentence,
)

def test_strip_diacritics_removes_hr_marks():
    assert strip_diacritics("škola čašica đak žaba ćup") == "skola casica djak zaba cup"

def test_typo_swap_changes_one_char():
    rng = random.Random(0)
    out = typo_swap("skola", rng)
    assert out != "skola"
    assert abs(len(out) - len("skola")) <= 1

def test_apply_confusion_uses_error_form():
    cs = {"škola": ["skola"]}
    rng = random.Random(0)
    assert apply_confusion("škola", cs, rng) == "skola"

def test_apply_confusion_returns_original_when_absent():
    rng = random.Random(0)
    assert apply_confusion("more", {}, rng) == "more"

def test_corrupt_sentence_is_deterministic_with_seed():
    cs = {"škola": ["skola"]}
    a = corrupt_sentence("Idem u škola danas", cs, seed=42)
    b = corrupt_sentence("Idem u škola danas", cs, seed=42)
    assert a == b

def test_corrupt_sentence_changes_something():
    cs = {}
    out = corrupt_sentence("Ovo je duga rečenica sa školom", cs, seed=1, p=1.0)
    assert out != "Ovo je duga rečenica sa školom"
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_noise.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `noise.py`**

```python
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
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_noise.py -v`
Expected: PASS (6 testova)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/noise.py tests/test_noise.py
git commit -m "feat: generatori sintetickih gresaka"
```

---

### Task 5: Preuzimanje sirovih izvora

Skripta koja preuzima hr500k (CLARIN.SI direktni linkovi) i daje upute za ispravi.me
(GitHub repo). Ispravi.me se klonira preko git-a. Funkcije su tanke omotnice oko HTTP/git,
pa test provjerava samo konstrukciju URL-ova i postojanje ciljne mape.

**Files:**
- Create: `src/crogrammar/data/download.py`
- Test: `tests/test_download.py`

- [ ] **Step 1: Napiši failing test**

```python
from crogrammar.data.download import HR500K_FILES, ispravime_repo_url, ensure_dir
from pathlib import Path

def test_hr500k_files_are_gz_urls():
    assert any("hr500k-train" in u for u in HR500K_FILES)
    assert all(u.startswith("https://") for u in HR500K_FILES)

def test_ispravime_repo_url():
    assert ispravime_repo_url().endswith(".git")
    assert "Ispravi-Me" in ispravime_repo_url()

def test_ensure_dir_creates(tmp_path):
    target = tmp_path / "raw" / "sub"
    ensure_dir(target)
    assert Path(target).is_dir()
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_download.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `download.py`**

```python
import subprocess
import urllib.request
from pathlib import Path

_BASE = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1792"

HR500K_FILES = [
    f"{_BASE}/hr500k-train.conllu.gz?sequence=3&isAllowed=y",
    f"{_BASE}/hr500k-dev.conllu.gz?sequence=4&isAllowed=y",
    f"{_BASE}/hr500k-test.conllu.gz?sequence=5&isAllowed=y",
]


def ispravime_repo_url() -> str:
    return "https://github.com/Ispravi-Me/Dataset-of-Misspelings-and-Corrections.git"


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_hr500k(raw_dir):
    d = ensure_dir(Path(raw_dir) / "hr500k")
    for url in HR500K_FILES:
        name = url.split("/")[-1].split("?")[0]
        dest = d / name
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
    return d


def clone_ispravime(raw_dir):
    d = Path(raw_dir) / "ispravime"
    if not d.exists():
        subprocess.run(["git", "clone", "--depth", "1", ispravime_repo_url(), str(d)], check=True)
    return d
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_download.py -v`
Expected: PASS (3 testa)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/download.py tests/test_download.py
git commit -m "feat: preuzimanje sirovih izvora (hr500k, ispravi.me)"
```

---

### Task 6: Orkestracija dataseta → JSONL

Spaja sve: čita čiste rečenice (hr500k + eventualno tekstualni korpus), primjenjuje
`corrupt_sentence` s confusion setom, generira parove `{"src","tgt"}`, dijeli na
train/dev/test. **Stvarni** ispravi.me rečenični parovi (gdje su dostupni cijeli
rečenični kontekst) idu prioritetno u dev/test.

**Files:**
- Create: `src/crogrammar/data/build_dataset.py`
- Test: `tests/test_build_dataset.py`

- [ ] **Step 1: Napiši failing test**

```python
import json
from crogrammar.data.build_dataset import make_pairs, split_pairs, write_jsonl

def test_make_pairs_creates_src_tgt():
    clean = ["Idem u školu", "On je otišao"]
    cs = {"školu": ["skolu"]}
    pairs = make_pairs(clean, cs, seed=7, variants=1)
    assert all(set(p.keys()) == {"src", "tgt"} for p in pairs)
    assert all(p["tgt"] in clean for p in pairs)

def test_make_pairs_variants_multiply():
    clean = ["Idem u školu"]
    pairs = make_pairs(clean, {}, seed=7, variants=3)
    assert len(pairs) == 3

def test_split_pairs_ratios():
    pairs = [{"src": str(i), "tgt": str(i)} for i in range(100)]
    train, dev, test = split_pairs(pairs, dev_frac=0.1, test_frac=0.1, seed=0)
    assert len(train) == 80 and len(dev) == 10 and len(test) == 10
    # bez preklapanja
    all_src = {p["src"] for p in train + dev + test}
    assert len(all_src) == 100

def test_write_jsonl_roundtrip(tmp_path):
    pairs = [{"src": "skola", "tgt": "škola"}]
    out = tmp_path / "d.jsonl"
    write_jsonl(pairs, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0]) == {"src": "skola", "tgt": "škola"}
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_build_dataset.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `build_dataset.py`**

```python
import json
import random
from pathlib import Path

from .noise import corrupt_sentence


def make_pairs(clean_sentences, confusion, seed: int, variants: int = 1):
    pairs = []
    for i, tgt in enumerate(clean_sentences):
        for v in range(variants):
            src = corrupt_sentence(tgt, confusion, seed=seed + i * 1000 + v)
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
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_build_dataset.py -v`
Expected: PASS (4 testa)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/data/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: orkestracija dataseta u JSONL (train/dev/test)"
```

---

## Faza 2 — Inference sučelje (s dummy modelom)

### Task 7: Segmentacija na rečenice

Omotnica oko `classla`, ali s čistim sučeljem i regex-fallbackom koji radi bez modela
(za testove). `classla` se učita lijeno samo kad je dostupan; testovi koriste fallback.

**Files:**
- Create: `src/crogrammar/inference/__init__.py` (prazan)
- Create: `src/crogrammar/inference/segment.py`
- Test: `tests/test_segment.py`

- [ ] **Step 1: Napiši failing test**

```python
from crogrammar.inference.segment import split_sentences

def test_splits_on_sentence_end():
    text = "Idem u školu. Vraćam se sutra! Zar ne?"
    assert split_sentences(text) == ["Idem u školu.", "Vraćam se sutra!", "Zar ne?"]

def test_single_sentence_no_terminator():
    assert split_sentences("Bok svima") == ["Bok svima"]

def test_empty_string():
    assert split_sentences("") == []

def test_preserves_diacritics():
    assert split_sentences("Čašica žeđi.") == ["Čašica žeđi."]
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_segment.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `segment.py`**

```python
import regex as re

_SENT_RE = re.compile(r"[^.!?]+[.!?]+|\S[^.!?]*$", re.UNICODE)


def split_sentences(text: str) -> list:
    if not text or not text.strip():
        return []
    return [m.group(0).strip() for m in _SENT_RE.finditer(text) if m.group(0).strip()]
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_segment.py -v`
Expected: PASS (4 testa)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/inference/__init__.py src/crogrammar/inference/segment.py tests/test_segment.py
git commit -m "feat: segmentacija teksta na recenice"
```

---

### Task 8: Diff → edits s tipizacijom

Uspoređuje originalnu i ispravljenu rečenicu, proizvodi listu izmjena s pozicijama u
**originalnom** tekstu i grubom klasifikacijom tipa. Koristi `difflib.SequenceMatcher`
na riječima, a pozicije mapira natrag na znakove.

**Files:**
- Create: `src/crogrammar/inference/diff.py`
- Test: `tests/test_diff.py`

- [ ] **Step 1: Napiši failing test**

```python
from crogrammar.inference.diff import compute_edits, classify_edit

def test_no_edits_when_identical():
    assert compute_edits("On je tu", "On je tu") == []

def test_single_word_replacement():
    edits = compute_edits("Idem u skolu", "Idem u školu")
    assert len(edits) == 1
    e = edits[0]
    assert e["original"] == "skolu"
    assert e["suggestion"] == "školu"
    assert "Idem u skolu"[e["start"]:e["end"]] == "skolu"

def test_classify_diacritic():
    assert classify_edit("skola", "škola") == "diacritic"

def test_classify_morphology():
    assert classify_edit("otiša", "otišao") == "morphology"

def test_classify_spelling_default():
    assert classify_edit("teik", "tekst") == "spelling"

def test_multiple_edits_positions():
    edits = compute_edits("on je otiša u skolu", "on je otišao u školu")
    originals = [e["original"] for e in edits]
    assert originals == ["otiša", "skolu"]
    for e in edits:
        assert "on je otiša u skolu"[e["start"]:e["end"]] == e["original"]
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_diff.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `diff.py`**

```python
import difflib

from ..data.noise import strip_diacritics


def classify_edit(original: str, suggestion: str) -> str:
    if strip_diacritics(suggestion) == original and suggestion != original:
        return "diacritic"
    if suggestion.startswith(original) or original.startswith(suggestion):
        return "morphology"
    return "spelling"


def _word_spans(text: str):
    spans = []
    idx = 0
    for word in text.split(" "):
        start = text.index(word, idx) if word else idx
        spans.append((word, start, start + len(word)))
        idx = start + len(word)
    return spans


def compute_edits(original: str, corrected: str) -> list:
    orig_spans = _word_spans(original)
    orig_words = [w for w, _, _ in orig_spans]
    corr_words = corrected.split(" ")
    sm = difflib.SequenceMatcher(a=orig_words, b=corr_words)
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        orig_text = " ".join(orig_words[i1:i2])
        sugg_text = " ".join(corr_words[j1:j2])
        if i1 < len(orig_spans):
            start = orig_spans[i1][1]
            end = orig_spans[i2 - 1][2] if i2 - 1 < len(orig_spans) and i2 > i1 else start
        else:
            start = end = len(original)
        edits.append({
            "start": start,
            "end": end,
            "original": orig_text,
            "suggestion": sugg_text,
            "type": classify_edit(orig_text, sugg_text),
        })
    return edits
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_diff.py -v`
Expected: PASS (6 testova)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/inference/diff.py tests/test_diff.py
git commit -m "feat: diff originala i ispravka u edits s tipizacijom"
```

---

### Task 9: Klasa `CroatianGEC` i `Result`

Sučelje jezgra. Prima `model_path` i objekt modela je apstrahiran preko `generate_fn`
callable-a (rečenica → ispravljena rečenica), da se može testirati bez pravog modela
(dummy identitet ili fiksne zamjene). U produkciji `from_pretrained` konstruira
`generate_fn` oko HuggingFace modela.

**Files:**
- Create: `src/crogrammar/inference/gec.py`
- Test: `tests/test_gec.py`

- [ ] **Step 1: Napiši failing test**

```python
from crogrammar.inference.gec import CroatianGEC, Result

def dummy_fix(sentence: str) -> str:
    return sentence.replace("skolu", "školu").replace("otiša", "otišao")

def test_correct_returns_result():
    gec = CroatianGEC(generate_fn=dummy_fix)
    res = gec.correct("on je otiša u skolu")
    assert isinstance(res, Result)
    assert res.corrected == "on je otišao u školu"

def test_correct_produces_edits():
    gec = CroatianGEC(generate_fn=dummy_fix)
    res = gec.correct("on je otiša u skolu")
    kinds = {e["type"] for e in res.edits}
    assert "diacritic" in kinds
    assert "morphology" in kinds

def test_correct_multi_sentence():
    gec = CroatianGEC(generate_fn=dummy_fix)
    res = gec.correct("Idem u skolu. On je otiša.")
    assert "školu" in res.corrected
    assert "otišao" in res.corrected

def test_correct_clean_text_no_edits():
    gec = CroatianGEC(generate_fn=lambda s: s)
    res = gec.correct("Sve je u redu.")
    assert res.edits == []
    assert res.corrected == "Sve je u redu."

def test_result_to_dict():
    gec = CroatianGEC(generate_fn=dummy_fix)
    res = gec.correct("skolu")
    d = res.to_dict()
    assert set(d.keys()) == {"corrected", "edits"}
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_gec.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `gec.py`**

```python
from dataclasses import dataclass, field

from .segment import split_sentences
from .diff import compute_edits


@dataclass
class Result:
    corrected: str
    edits: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"corrected": self.corrected, "edits": self.edits}


class CroatianGEC:
    def __init__(self, model_path: str = None, generate_fn=None, max_len: int = 256):
        self.max_len = max_len
        if generate_fn is not None:
            self._generate = generate_fn
        elif model_path is not None:
            self._generate = self._build_hf_generate(model_path)
        else:
            raise ValueError("Zadaj model_path ili generate_fn")

    def _build_hf_generate(self, model_path):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()

        def _gen(sentence: str) -> str:
            inputs = tok("ispravi: " + sentence, return_tensors="pt",
                         truncation=True, max_length=self.max_len)
            out = model.generate(**inputs, max_length=self.max_len)
            return tok.decode(out[0], skip_special_tokens=True)

        return _gen

    def correct(self, text: str) -> Result:
        sentences = split_sentences(text)
        corrected_sentences = [self._generate(s) for s in sentences]
        corrected = " ".join(corrected_sentences)
        edits = compute_edits(text, corrected) if sentences else []
        return Result(corrected=corrected, edits=edits)
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_gec.py -v`
Expected: PASS (5 testova)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/inference/gec.py tests/test_gec.py
git commit -m "feat: CroatianGEC sucelje s correct() i Result"
```

---

## Faza 3 — Evaluacija

### Task 10: GLEU i P/R/F0.5 metrike

GLEU preko `sacrebleu` (koji ima GLEU? — koristimo vlastitu n-gram GLEU implementaciju
za GEC prema Napoles 2015, jer je to standard). Dodatno jednostavan word-level
precision/recall/F0.5 nad edit skupovima.

**Files:**
- Create: `src/crogrammar/eval/__init__.py` (prazan)
- Create: `src/crogrammar/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Napiši failing test**

```python
from crogrammar.eval.metrics import gleu_sentence, prf_edits

def test_gleu_perfect_is_one():
    assert gleu_sentence("on je otišao u školu", "on je otišao u školu") == 1.0

def test_gleu_worse_is_lower():
    good = gleu_sentence("on je otišao u školu", "on je otišao u školu")
    bad = gleu_sentence("on je otiša u skolu", "on je otišao u školu")
    assert bad < good

def test_prf_all_correct():
    gold = {("skolu", "školu"), ("otiša", "otišao")}
    pred = {("skolu", "školu"), ("otiša", "otišao")}
    p, r, f = prf_edits(pred, gold)
    assert (p, r) == (1.0, 1.0)
    assert abs(f - 1.0) < 1e-9

def test_prf_partial():
    gold = {("skolu", "školu"), ("otiša", "otišao")}
    pred = {("skolu", "školu")}
    p, r, f = prf_edits(pred, gold)
    assert p == 1.0
    assert r == 0.5
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `metrics.py`**

```python
from collections import Counter


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


def prf_edits(pred: set, gold: set, beta: float = 0.5):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    if p == 0 and r == 0:
        return p, r, 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0
    return p, r, f
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (4 testa)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/eval/__init__.py src/crogrammar/eval/metrics.py tests/test_metrics.py
git commit -m "feat: GLEU i P/R/F0.5 metrike"
```

---

## Faza 4 — Trening

### Task 11: Konfiguracija treninga

Dataclass s hiperparametrima; jedan izvor istine za trening notebook i skriptu.

**Files:**
- Create: `src/crogrammar/train/__init__.py` (prazan)
- Create: `src/crogrammar/train/config.py`
- Test: `tests/test_train_config.py`

- [ ] **Step 1: Napiši failing test**

```python
from crogrammar.train.config import TrainConfig

def test_defaults():
    cfg = TrainConfig()
    assert cfg.base_model == "google/byt5-small"
    assert cfg.task_prefix == "ispravi: "
    assert cfg.max_source_len > 0
    assert cfg.seed == 42

def test_override():
    cfg = TrainConfig(num_epochs=5, batch_size=8)
    assert cfg.num_epochs == 5
    assert cfg.batch_size == 8

def test_as_dict_serializable():
    import json
    cfg = TrainConfig()
    json.dumps(cfg.as_dict())  # ne smije baciti
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_train_config.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `config.py`**

```python
from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    base_model: str = "google/byt5-small"
    task_prefix: str = "ispravi: "
    train_file: str = "data/processed/train.jsonl"
    dev_file: str = "data/processed/dev.jsonl"
    output_dir: str = "models/byt5-hr-gec"
    max_source_len: int = 256
    max_target_len: int = 256
    batch_size: int = 4
    grad_accum: int = 8
    num_epochs: int = 3
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    save_steps: int = 1000
    eval_steps: int = 1000
    seed: int = 42
    fp16: bool = True

    def as_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_train_config.py -v`
Expected: PASS (3 testa)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/train/__init__.py src/crogrammar/train/config.py tests/test_train_config.py
git commit -m "feat: TrainConfig hiperparametri"
```

---

### Task 12: Trening skripta (Seq2SeqTrainer)

Funkcija `build_datasets(tokenizer, cfg)` (testabilna s malim JSONL-om) i `train(cfg)`
koji koristi `Seq2SeqTrainer`. Testiramo samo tokenizaciju/mapiranje dataseta na malom
uzorku (bez GPU treninga). `train()` podržava `resume_from_checkpoint`.

**Files:**
- Create: `src/crogrammar/train/train.py`
- Test: `tests/test_train_prepare.py`

- [ ] **Step 1: Napiši failing test**

```python
import json
from crogrammar.train.train import preprocess_batch

class FakeTok:
    pad_token_id = 0
    def __call__(self, texts, max_length=None, truncation=True, padding=False):
        return {"input_ids": [[len(t)] for t in texts],
                "attention_mask": [[1] for _ in texts]}

def test_preprocess_batch_adds_prefix_and_labels():
    batch = {"src": ["skolu", "otiša"], "tgt": ["školu", "otišao"]}
    tok = FakeTok()
    out = preprocess_batch(batch, tok, prefix="ispravi: ", max_src=32, max_tgt=32)
    assert "input_ids" in out
    assert "labels" in out
    assert len(out["input_ids"]) == 2
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_train_prepare.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `train.py`**

```python
def preprocess_batch(batch, tokenizer, prefix, max_src, max_tgt):
    sources = [prefix + s for s in batch["src"]]
    model_inputs = tokenizer(sources, max_length=max_src, truncation=True)
    labels = tokenizer(batch["tgt"], max_length=max_tgt, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train(cfg, resume_from_checkpoint=None):
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model)

    data = load_dataset("json", data_files={"train": cfg.train_file, "dev": cfg.dev_file})
    tokenized = data.map(
        lambda b: preprocess_batch(b, tokenizer, cfg.task_prefix,
                                   cfg.max_source_len, cfg.max_target_len),
        batched=True, remove_columns=data["train"].column_names,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.num_epochs,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=cfg.fp16,
        seed=cfg.seed,
        logging_steps=100,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=tokenized["train"], eval_dataset=tokenized["dev"],
        data_collator=collator, tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_train_prepare.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/train/train.py tests/test_train_prepare.py
git commit -m "feat: trening skripta (Seq2SeqTrainer) + preprocess"
```

---

### Task 13: Colab/Kaggle notebook + README

Notebook koji: montira Drive, preuzima izvore, gradi dataset, poziva `train(cfg)`,
sprema artefakt na Drive, evaluira. README dokumentira upotrebu, licence i atribucije.

**Files:**
- Create: `notebooks/train_colab.ipynb`
- Create: `README.md`

- [ ] **Step 1: Napiši `README.md`**

```markdown
# Crogrammar

Hrvatski gramaticki checker (GEC) — neuronski seq2seq model (ByT5 fine-tuned).

## Instalacija
    pip install -e ".[dev]"

## Priprema podataka
    python -m crogrammar.data.download          # hr500k + ispravi.me
    python -m crogrammar.data.build_dataset     # -> data/processed/*.jsonl

## Trening (Colab/Kaggle)
Otvori `notebooks/train_colab.ipynb`.

## Upotreba
    from crogrammar.inference.gec import CroatianGEC
    gec = CroatianGEC(model_path="models/byt5-hr-gec")
    print(gec.correct("on je otiša u skolu").to_dict())

## Testovi
    pytest

## Licence i atribucije
- ispravi.me dataset: Gledec et al. (2023), CC BY-NC-SA 4.0 (nekomercijalno).
- hr500k: Ljubešić & Samardžić, CLARIN.SI, CC BY-SA 4.0.
- Model je izveden iz CC BY-NC-SA podataka → nekomercijalna upotreba.
```

- [ ] **Step 2: Kreiraj notebook skeleton**

Kreiraj `notebooks/train_colab.ipynb` s ćelijama (markdown + code) redom:
1. `!pip install -e ".[dev]"` (nakon clonanja repo-a)
2. `from google.colab import drive; drive.mount('/content/drive')`
3. `python -m crogrammar.data.download`
4. `python -m crogrammar.data.build_dataset`
5. ```python
   from crogrammar.train.config import TrainConfig
   from crogrammar.train.train import train
   cfg = TrainConfig(output_dir="/content/drive/MyDrive/crogrammar/byt5-hr-gec")
   train(cfg, resume_from_checkpoint=True)
   ```
6. Evaluacijska ćelija koja učita model i izračuna GLEU na test setu.

- [ ] **Step 3: Provjeri da je notebook valjan JSON**

Run: `python -c "import json,sys; json.load(open('notebooks/train_colab.ipynb', encoding='utf-8')); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add README.md notebooks/train_colab.ipynb
git commit -m "docs: README i Colab trening notebook"
```

---

### Task 14: End-to-end evaluacijska skripta

CLI ulaz koji učita artefakt modela i JSONL test set, izračuna korpusni GLEU, ispiše
rezultat. Testira se s dummy `generate_fn` (bez pravog modela).

**Files:**
- Create: `src/crogrammar/eval/run_eval.py`
- Test: `tests/test_run_eval.py`

- [ ] **Step 1: Napiši failing test**

```python
from crogrammar.eval.run_eval import evaluate_pairs

def test_evaluate_pairs_perfect():
    pairs = [{"src": "skolu", "tgt": "školu"}]
    fix = lambda s: "školu"
    score = evaluate_pairs(pairs, fix)
    assert score["gleu"] == 1.0

def test_evaluate_pairs_reports_count():
    pairs = [{"src": "a", "tgt": "a"}, {"src": "b", "tgt": "b"}]
    score = evaluate_pairs(pairs, lambda s: s)
    assert score["n"] == 2
```

- [ ] **Step 2: Pokreni test — mora pasti**

Run: `pytest tests/test_run_eval.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implementiraj `run_eval.py`**

```python
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
```

- [ ] **Step 4: Pokreni test — mora proći**

Run: `pytest tests/test_run_eval.py -v`
Expected: PASS (2 testa)

- [ ] **Step 5: Commit**

```bash
git add src/crogrammar/eval/run_eval.py tests/test_run_eval.py
git commit -m "feat: end-to-end evaluacija (korpusni GLEU)"
```

---

## Završna provjera

### Task 15: Puni test suite + integracija

- [ ] **Step 1: Pokreni sve testove**

Run: `pytest -v`
Expected: svi testovi prolaze (Task 2–14).

- [ ] **Step 2: Integracijska provjera inference lanca s dummy modelom**

Run:
```bash
python -c "from crogrammar.inference.gec import CroatianGEC; g=CroatianGEC(generate_fn=lambda s: s.replace('skolu','školu')); print(g.correct('Idem u skolu.').to_dict())"
```
Expected: ispisan dict s `corrected` i jednom `diacritic` izmjenom.

- [ ] **Step 3: Commit (ako je bilo popravaka)**

```bash
git add -A
git commit -m "test: puni suite prolazi, integracija inference lanca"
```

---

## Self-Review (autor plana)

- **Spec coverage:** [1] pipeline → Task 2–6; [2] trening → Task 11–13; [3] artefakt → Task 12–13 (save_model + METADATA u notebooku); [4] inference → Task 7–9; eval → Task 10, 14; testiranje → svi taskovi + Task 15; struktura → Task 1. Pokriveno.
- **Placeholder scan:** svi kod-koraci imaju konkretan kod; nema TODO/TBD.
- **Type consistency:** `Result.to_dict()`, `compute_edits` vraća dict s ključevima `start/end/original/suggestion/type` — dosljedno korišteno u Task 8, 9. `TrainConfig.task_prefix` = `"ispravi: "` dosljedno s `_build_hf_generate` i `preprocess_batch`.
- **Napomena:** METADATA.json (verzija/GLEU/licenca) iz speca [3] generira se u evaluacijskoj ćeliji notebooka; ako se želi automatizirati, dodati mali helper u budućoj iteraciji (YAGNI zasad).
