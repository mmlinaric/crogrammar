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
