# Crogrammar

Hrvatski gramaticki checker (GEC) — neuronski seq2seq model (ByT5 fine-tuned).

Otkriva i ispravlja gramaticke, pravopisne, morfoloske i vokabularne greske u
hrvatskom tekstu. Model je prenosiv artefakt (HuggingFace format) koji se poziva
preko jednog cistog sucelja `CroatianGEC.correct(text)`.

## Instalacija

Lokalni razvoj i testovi (bez GPU-a, samo `regex` + standardna knjiznica):

    pip install -e ".[dev]"

Za trening (na Colab/Kaggle) instaliraj i teske ovisnosti:

    pip install -e ".[train]"

## Priprema podataka

    python -m crogrammar.data.download          # hr500k + ispravi.me u data/raw/
    python -m crogrammar.data.build_dataset     # -> data/processed/*.jsonl

Izvori:
- **hr500k** (CLARIN.SI, CC BY-SA 4.0) — cist tekst + morfoloske oznake.
- **hunspell-hr** (`krunose/hunspell-hr`) — rjecnik hrvatskih rijeci (~15k s
  dijakritikom) iz kojeg se gradi dijakriticki confusion set za generiranje gresaka.

> Napomena: originalni ispravi.me dataset (33M parova, CC BY-NC-SA 4.0) vise nije
> javno dostupan (GitHub org `Ispravi-Me` je ispraznjen), pa umjesto njega koristimo
> hunspell-hr rjecnik. Kod za ispravi.me (`load_confusion_from_dir`, `clone_ispravime`)
> ostaje u projektu za slucaj da dataset ponovno postane dostupan.

### Stvarni test set (RAPUT)

Osim sintetickog test seta, model se evaluira i na **RAPUT 1.0** — stvarnom,
rucno anotiranom korpusu nesavrsenog hrvatskog pisanja (34k recenica, parovi
napisano→ispravljeno). To daje *pravi* GLEU, ne samo sinteticki. Licenca CC BY-SA 4.0.
Preuzima se preko `crogrammar.data.download.download_raput`.

## Trening (Colab/Kaggle)

Otvori `notebooks/train_colab.ipynb`. Postupak:
1. Montiraj Google Drive (za nastavljive checkpointe).
2. Preuzmi izvore i izgradi dataset.
3. Fine-tune ByT5 (`google/byt5-small`) preko `crogrammar.train.train.train(cfg)`.
4. Spremi artefakt na Drive i evaluiraj (GLEU) na test setu.

Besplatni Colab (Tesla T4, 16 GB) ili Kaggle su dovoljni za `byt5-small`.

## Upotreba

```python
from crogrammar.inference.gec import CroatianGEC

gec = CroatianGEC(model_path="models/byt5-hr-gec")
res = gec.correct("on je otiša u skolu")
print(res.to_dict())
# {
#   "corrected": "on je otišao u školu",
#   "edits": [
#     {"start": 6, "end": 11, "original": "otiša", "suggestion": "otišao", "type": "morphology"},
#     {"start": 14, "end": 19, "original": "skolu", "suggestion": "školu", "type": "diacritic"}
#   ]
# }
```

Za testiranje bez modela, `CroatianGEC` prima i `generate_fn` (rečenica → ispravak):

```python
gec = CroatianGEC(generate_fn=lambda s: s.replace("skolu", "školu"))
```

## Struktura

    src/crogrammar/
      data/         # pipeline: conllu, confusion, noise, download, build_dataset
      inference/    # segment, diff, gec (CroatianGEC + Result)
      train/        # config (TrainConfig), train (Seq2SeqTrainer)
      eval/         # metrics (GLEU, P/R/F0.5), run_eval
    notebooks/      # Colab/Kaggle trening
    tests/          # unit testovi (bez GPU-a)

## Testovi

    pytest

## Licence i atribucije

- **ispravi.me dataset**: Gledec, Horvat, Mikuc, Blaskovic (2023),
  *A Comprehensive Dataset of Spelling Errors and Users' Corrections in Croatian
  Language*, Data 8(5):89. Licenca **CC BY-NC-SA 4.0** (nekomercijalno).
- **hr500k**: Ljubesic & Samardzic, CLARIN.SI, **CC BY-SA 4.0**.

Model je izveden iz CC BY-NC-SA podataka, pa naslijeduje **nekomercijalno**
ogranicenje. Za komercijalnu upotrebu potreban je dogovor s autorima ispravi.me
dataseta.
