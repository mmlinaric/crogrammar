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
