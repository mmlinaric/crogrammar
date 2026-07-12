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
