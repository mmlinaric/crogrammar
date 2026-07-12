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
