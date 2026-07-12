# Crogrammar v0.2.0 — Veliki upgrade modela — Dizajn

**Datum:** 2026-07-12
**Status:** Odobreno (spremno za planiranje implementacije)

## Cilj

Znatno bolji GEC model za hrvatski (v0.2.0) u odnosu na v0.1.0. Pet nadogradnji:
veći model, više i raznovrsniji podaci, realnije greške (uklj. stvarne RAPUT greške),
više epoha, te pošteno mjerenje napretka ručnim test setom. Trening se seli na Kaggle
(rješava Google Drive storage problem).

## Kontekst (v0.1.0 baseline)

- ByT5-small, 1 epoha, sintetičke greške (dijakritika, tipfeleri, ao/io, padeži),
  hunspell-hr confusion set, hr500k čist tekst.
- Rezultati: GLEU 0.957 (sintetički), **~0.45 (RAPUT stvarni)**, ~0.65 (dijakritika-neutralno).
- Postoji: homograf-safe dijakritika, DirectML/CPU inference, RAPUT parser, batch eval.

## Ključne odluke

| Odluka | Izbor | Obrazloženje |
|--------|-------|--------------|
| Platforma | **Kaggle** (P100 16GB, 30h/tj) | Neovisno o Drive/Colab limitu; 20GB output |
| Storage | `/kaggle/working` + Kaggle Dataset za resume | Nema Drive kvote |
| Model | **ByT5-base** (~580M) | Najbolji za HR dijakritiku/morfologiju |
| Čist tekst | **hr500k + hrvatska Wikipedia** | Širina vokabulara + morfološke oznake |
| Greške | Prošireni generator + **RAPUT stvarni parovi u trening** | Realizam |
| RAPUT | **Sav u trening** (pretežan ×3-5) | Model uči prave greške |
| Mjerenje | **Ručni test set** (~60-100 rečenica) | RAPUT više nije dostupan kao test |
| Epohe | **3** | Bolja konvergencija |
| Preciznost | **fp32** | ByT5 + fp16 = NaN (potvrđeno na v0.1.0) |

## Arhitektura / tok

```
KAGGLE notebook:
  [preuzmi: hr500k, hunspell-hr, RAPUT, Wikipedia(HF)]
    -> [ocisti Wikipediju -> ~1-2M recenica]
    -> [generiraj sinteticke parove + RAPUT stvarni parovi (pretezani)]
    -> [fine-tune ByT5-base, 3 epohe, fp32, resume-safe]
    -> [spremi u /kaggle/working + kao Kaggle Dataset]
    -> [evaluiraj na rucnom test setu: GLEU po kategoriji]
LOKALNO (RX570 / DirectML):
  [skini model] -> [CroatianGEC.correct() / testiranje]
```

## Komponente

### [1] Wikipedia pipeline — `src/crogrammar/data/wikipedia.py` (novo)
- `is_clean_sentence(text) -> bool` — filtar kvalitete: duljina 20-200 znakova, sadrži
  slova, odbija previše brojki/simbola/URL, wiki-ostatke (`==`, `[[`, `{{`, `|`).
- `load_wiki_sentences(limit=None)` — učita `wikimedia/wikipedia` config `20231101.hr`
  preko HF `datasets`, segmentira člance (`split_sentences`), filtrira, deduplicira.
- Lijeni import `datasets` (samo za trening okruženje).
- Testira se filtar na uzorcima (bez mreže).

### [2] Prošireni generator grešaka — `src/crogrammar/data/noise.py`
Dodati (uz postojeće):
- `merge_words(w1, w2)` — spajanje (`u školu` -> `uškolu`).
- `split_word(word, rng)` — razdvajanje na valjanoj granici (`nemogu` -> `ne mogu`).
- `punctuation_noise(sentence, rng)` — brisanje/dodavanje zareza/točke.
- `case_noise(word, rng)` — nasumično veliko/malo početno slovo.
- `corrupt_sentence` proširen da uključi nove tipove (i dalje homograf-safe za dijakritiku).

### [3] RAPUT training pairs — `src/crogrammar/data/raput.py`
- `raput_training_pairs(path)` — vrati parove `(orig, form)` gdje se razlikuju
  (ignorirajući čisto početno veliko slovo), kao `{"src","tgt"}`.

### [4] Miješanje izvora — `src/crogrammar/data/build_dataset.py`
- `mix_sources(synthetic_pairs, real_pairs, real_weight=4)` — spoji, preteži stvarne
  parove (ponovi `real_weight` puta), promiješaj.
- `make_pairs` već postoji; proširiti orkestraciju za više izvora čistog teksta.

### [5] Config profil — `src/crogrammar/train/config.py`
Nove default vrijednosti ostaju kompatibilne; Kaggle notebook konstruira:
`TrainConfig(base_model="google/byt5-base", batch_size=4, grad_accum=8,
max_source_len=160, max_target_len=160, num_epochs=3, fp16=False)`.
Ne mijenjamo postojeće testove configa (samo dodajemo/koristimo parametre).

### [6] Ručni test set — `tests/manual_test_set.jsonl` (novo)
~60-100 rečenica, svaki redak `{"src","tgt","kategorija"}`, kategorije: diacritic,
morphology (ao/io, padež), split_merge, punctuation, homograph_keep, case.
Homograf primjeri (`imam sto kuna`) imaju `src == tgt` (ne smije dirati).

### [7] Kaggle notebook — `notebooks/train_kaggle.ipynb` (novo)
Ćelije: install `-e .[train]`, preuzmi izvore, build Wikipedia+hr500k+RAPUT dataset,
train (resume iz Kaggle Dataseta ako postoji), spremi model + kao Dataset, eval na
ručnom setu (GLEU po kategoriji + usporedba sa starim ako je dostupan).

## Testiranje (TDD, lokalno bez GPU-a)

- `is_clean_sentence` — prihvaća dobre, odbija wiki-smeće/prekratko/predugo.
- Novi generatori — determinizam sa seedom, ispravno ponašanje.
- `raput_training_pairs` — ispravna ekstrakcija, ignorira case-only.
- `mix_sources` — pretezanje i brojevi.
- Ručni test set — valjan JSONL, sve kategorije prisutne.
- Model: evaluacija na ručnom setu (ne unit test).

## Evaluacija / mjerenje napretka

- Isti ručni test set za v0.1.0 i v0.2.0.
- Metrike: GLEU ukupno + GLEU po kategoriji + dijakritika-neutralni GLEU.
- Cilj: v0.2.0 > v0.1.0 na ukupnom i na morfologiji/split_merge.

## Redoslijed implementacije

1. Prošireni generatori grešaka (`noise.py`) + testovi.
2. `wikipedia.py` filtar + loader + testovi.
3. `raput_training_pairs` + `mix_sources` + testovi.
4. ByT5-base config korištenje (notebook) — bez lomljenja postojećih testova.
5. Ručni test set (`manual_test_set.jsonl`).
6. `notebooks/train_kaggle.ipynb`.
7. Puni test suite + lokalna evaluacija starog modela na ručnom setu (baseline broj).

## Ostaje izvan koda (korisnik/okruženje)

- Trening na Kaggle GPU (P100), 2-3 sesije s resume.
- Download finalnog modela ili korištenje kao Kaggle Dataset.
- Lokalni inference na RX570 (DirectML) već radi.

## Atribucije (licence)

- hr500k: Ljubešić & Samardžić, CLARIN.SI, CC BY-SA 4.0.
- hunspell-hr: Denis Lacković, LGPL/SISSL.
- RAPUT 1.0: Kuvač Kraljević et al., CLARIN.SI, CC BY-SA 4.0.
- Wikipedia (hr): CC BY-SA 4.0.
- ByT5: Apache 2.0.
Svi izvori komercijalno-kompatibilni (uz atribuciju + ShareAlike na izvedeni dataset).
