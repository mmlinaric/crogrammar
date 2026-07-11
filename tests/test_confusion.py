from crogrammar.data.confusion import (
    build_confusion_set,
    read_hunspell_dic,
    build_confusion_from_wordlist,
)

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


DIC_SAMPLE = """5
škola/360
čašica
đak
more
a-mol/273
"""


def test_read_hunspell_dic_strips_flags():
    words = list(read_hunspell_dic(DIC_SAMPLE))
    assert "škola" in words
    assert "čašica" in words
    assert "a-mol" in words

def test_read_hunspell_dic_skips_count_line():
    words = list(read_hunspell_dic(DIC_SAMPLE))
    assert "5" not in words

def test_wordlist_confusion_maps_diacritic_word_to_stripped():
    cs = build_confusion_from_wordlist(["škola", "čašica", "đak"])
    assert cs["škola"] == ["skola"]
    assert cs["čašica"] == ["casica"]
    assert cs["đak"] == ["djak"]

def test_wordlist_confusion_skips_words_without_diacritics():
    cs = build_confusion_from_wordlist(["more", "kuca", "škola"])
    assert "more" not in cs
    assert "kuca" not in cs
    assert "škola" in cs
