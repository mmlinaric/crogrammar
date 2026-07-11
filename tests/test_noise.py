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
