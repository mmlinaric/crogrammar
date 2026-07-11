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
