import random
from crogrammar.data.noise import (
    strip_diacritics,
    strip_diacritics_safe,
    typo_swap,
    apply_confusion,
    contract_ao,
    change_case_ending,
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


def test_contract_ao_drops_final_o():
    assert contract_ao("otišao") == "otiša"
    assert contract_ao("radio") == "radi"
    assert contract_ao("rekao") == "reka"

def test_contract_ao_leaves_non_ao_words():
    assert contract_ao("kuća") == "kuća"
    assert contract_ao("stol") == "stol"
    assert contract_ao("auto") == "auto"

def test_change_case_ending_replaces_final_vowel():
    rng = random.Random(0)
    out = change_case_ending("kuća", rng)
    assert out != "kuća"
    assert out[:-1] == "kuć"
    assert out[-1] in "eiou"

def test_change_case_ending_leaves_consonant_ending():
    rng = random.Random(0)
    assert change_case_ending("stol", rng) == "stol"


def test_strip_diacritics_safe_skips_real_word_homographs():
    real_words = {"sto", "što", "posto", "pošto"}
    assert strip_diacritics_safe("što", real_words) == "što"
    assert strip_diacritics_safe("pošto", real_words) == "pošto"

def test_strip_diacritics_safe_strips_when_not_real_word():
    real_words = {"škola", "sto"}
    assert strip_diacritics_safe("škola", real_words) == "skola"

def test_strip_diacritics_safe_no_wordset_behaves_like_strip():
    assert strip_diacritics_safe("što", None) == "sto"
    assert strip_diacritics_safe("škola", set()) == "skola"

def test_corrupt_sentence_preserves_homograph_with_wordset():
    real_words = {"sto", "što", "jabuka", "jabuke"}
    out = corrupt_sentence("sto jabuka", {}, seed=3, p=1.0, real_words=real_words)
    assert "što" not in out.split()
