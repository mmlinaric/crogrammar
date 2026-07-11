# Crogrammar — Hrvatski gramatički checker (GEC) — Dizajn

**Datum:** 2026-07-11
**Status:** Odobreno (spremno za planiranje implementacije)

## Cilj

Izgraditi gramatički checker za **hrvatski jezik**, inspiriran servisom ispravi.me,
koji otkriva i ispravlja gramatičke, pravopisne, morfološke i vokabularne greške.

Središnji cilj je **prenosiv model** (artefakt) koji se kasnije može ugraditi u
bilo kakav pogon (CLI, REST API, web) preko jednog čistog sučelja. Korisnik ne traži
gotov UI zasad — traži model koji radi bilo gdje.

## Ključne odluke

| Odluka | Izbor | Obrazloženje |
|--------|-------|--------------|
| Jezik | Samo hrvatski | Fokusiran opseg |
| Metoda | Neuronski seq2seq GEC (fine-tuned transformer) | SOTA za morfološki bogate jezike (2025); nadmašuje gola pravila i goli LLM API |
| Temeljni model | **ByT5-small** (primarni), mT5-small (baseline) | Byte-level rješava dijakritiku (č/ć/đ/š/ž), bogatu morfologiju i OOV prirodno |
| Hardver | Google Colab / Kaggle (T4 16GB) | Dovoljno za fine-tuning modela ~300M; uz LoRA i veći |
| Opseg grešaka | Puni GEC (sve u jednom prolazu) | Pravopis + gramatika + slaganje + vokabular zajedno |
| Format izlaza | Ispravljeni tekst + diff (edits) | Model vraća čist tekst; izmjene računamo diffom naknadno |
| Licenca | Nekomercijalno (CC BY-NC-SA 4.0) | Zbog ispravi.me NC klauzule; koristimo sve izvore |

## Arhitektura

Četiri neovisne cjeline povezane jasnim sučeljima:

```
[1] Podatkovni pipeline  →  [2] Trening  →  [3] Model (artefakt)  →  [4] Inference sučelje
   (priprema parova)         (fine-tune)     (ByT5 težine)          correct(text) -> diff
```

Model je centralni artefakt. Sve prije njega ga proizvodi; sve poslije samo poziva
`correct(text)`. Time je model neovisan o pogonu i doista prenosiv.

## [1] Podatkovni pipeline

Kombiniramo dva izvora:

**A) Stvarni podaci — ispravi.me dataset**
- Izvor: `github.com/Ispravi-Me/Dataset-of-Misspelings-and-Corrections`
- 33M parova "greška → ispravak", 5.5M jedinstvenih, gzip TSV po godini.
- Licenca: **CC BY-NC-SA 4.0** (navođenje autora, nekomercijalno, ShareAlike).
- Parovi su na razini **riječi**, koristimo ih:
  1. kao **confusion set** (realne zamjene) za sintetičko generiranje,
  2. za direktne kratke primjere gdje kontekst nije nužan.

**B) Sintetički podaci — ubacivanje grešaka u čist tekst**
- Čist tekst: **hr500k** (CLARIN.SI, CC BY-SA 4.0, CoNLL-U s morfološkim oznakama)
  + Wikipedia/hrWaC.
- Ubacivanje grešaka pomoću:
  - confusion seta iz ispravi.me (realne zamjene),
  - dijakritičkih grešaka (č↔c, š↔s, ž↔z, ć↔c, đ↔dj),
  - morfoloških perturbacija (pogrešan padež/rod/broj, koristeći hr500k oznake),
  - tipografskih grešaka (zamjena/brisanje/umetanje susjednih slova).

**Output:** `train.jsonl`, `dev.jsonl`, `test.jsonl`, redak:
```json
{"src": "on je otiša u skolu", "tgt": "on je otišao u školu"}
```

**Test set mora biti stvaran, ne sintetički** — dio ispravi.me / ručno provjerenih
rečenica držimo odvojeno za evaluaciju prave izvedbe.

## [2] Trening

**Dvije faze (standard za GEC):**
1. **Pretrening na sintetičkim greškama** — velika količina (izvor B), opća gramatika ispravljanja.
2. **Fine-tuning na stvarnim greškama** — manja kvalitetna količina (ispravi.me), prilagodba pravim greškama.

**Format ulaza (T5 konvencija):**
```
ispravi: on je otiša u skolu   →   on je otišao u školu
```

**Postavke:**
- Rad na razini rečenice (ByT5 spor na dugim nizovima; max ~256–512 bajtova).
- Batch size prilagođen T4 + gradient accumulation, mixed precision (fp16/bf16).
- Early stopping po GLEU/ERRANT na dev setu.
- Checkpointi na Google Drive; trening **nastavljiv** (resume from checkpoint).

**Evaluacija:**
- GLEU (standard, BHASHA/MultiGEC 2025).
- ERRANT-stil precision/recall/F0.5 na stvarnom test setu.
- Cilj: usporedivo s literaturom (80+ GLEU).

**mT5-small kao baseline** — isti pipeline, usporedba na hrvatskom (YAGNI: samo ako
ostane vremena; ByT5 je primarni).

## [3] Model — artefakt

```
model/
  config.json          # arhitektura
  model.safetensors    # težine
  tokenizer + spiece   # ByT5 byte-level, minimalno
  METADATA.json        # verzija, datum, GLEU rezultat, licenca (CC BY-NC-SA)
```
Standardni HuggingFace format → radi svugdje gdje ima `transformers`; kasnije
konvertibilan u ONNX/GGUF.

## [4] Inference sučelje

Jedno čisto sučelje, jezgra svega:

```python
class CroatianGEC:
    def __init__(self, model_path): ...          # učita model jednom
    def correct(self, text: str) -> Result: ...  # glavna metoda
```

`correct(text)` iznutra:
1. Segmentacija teksta na rečenice (classla/syntok za hrvatski).
2. Inference ByT5 po rečenici (batch).
3. Spajanje natrag u cijeli tekst.
4. Diff — usporedba original vs. ispravak → lista izmjena.

**Izlazna struktura `Result`:**
```json
{
  "corrected": "On je otišao u školu.",
  "edits": [
    {"start": 6, "end": 11, "original": "otiša", "suggestion": "otišao", "type": "morphology"},
    {"start": 15, "end": 20, "original": "skolu", "suggestion": "školu", "type": "diacritic"}
  ]
}
```
- `edits` = token-level diff (difflib / ERRANT-stil poravnanje).
- `type` = gruba klasifikacija (diacritic / spelling / morphology / other) za budući UI.
- `start`/`end` = pozicije u **originalnom** tekstu za precizno isticanje.

Ovo sučelje je jedina točka koju CLI/API kasnije zovu.

## Struktura projekta

```
crogrammar/
  data/
    raw/              # preuzeti izvori (ispravi.me, hr500k) — .gitignore
    processed/        # train/dev/test .jsonl
  src/crogrammar/
    data/             # [1] pipeline: preuzimanje, confusion set, sintetičke greške
    train/            # [2] trening skripte + config
    inference/        # [4] CroatianGEC, segmentacija, diff, tipizacija
    eval/             # GLEU / ERRANT-stil metrike
  notebooks/          # Colab/Kaggle trening notebook
  models/             # [3] artefakt (.gitignore) + METADATA
  tests/              # unit testovi
  pyproject.toml
  README.md
```

## Testiranje

- **Deterministički dijelovi (unit testovi, bez GPU-a):** generatori sintetičkih
  grešaka, diff→edits logika, tipizacija, segmentacija rečenica, parsiranje CoNLL-U.
- **Model:** evaluira se na test setu (GLEU/F0.5), ne unit testom.
- **Sučelje `correct()`:** testira se s dummy modelom (identitet) da pipeline radi bez treninga.

## Redoslijed implementacije

1. Skele projekta + `pyproject.toml` + osnovni testovi.
2. **[1]** Podatkovni pipeline: preuzimanje → parsiranje → confusion set → sintetičke greške → `.jsonl`. *(testovi)*
3. **[4]** Inference sučelje s dummy modelom: segmentacija + diff + tipizacija. *(testovi)*
4. **[2]** Trening notebook: fine-tune ByT5, checkpointi na Drive, resume.
5. **[eval]** GLEU/ERRANT metrike + evaluacija.
6. Prvi pravi trening → artefakt [3] → spoji s [4] → provjeri na stvarnom tekstu.

Software (koraci 1,3,4,5) razvija se i testira bez GPU-a; GPU treba tek u koraku 4.

## Tech stack

Python, HuggingFace `transformers` / `datasets`, `classla` (hrvatski NLP),
`sacrebleu` / GLEU, `pytest`.

## Atribucije (obavezno zbog licenci)

- ispravi.me dataset: Gledec, Horvat, Mikuc, Blašković (2023), CC BY-NC-SA 4.0.
- hr500k: Ljubešić, Samardžić, CLARIN.SI, CC BY-SA 4.0.
