from crogrammar.eval.metrics import gleu_sentence, prf_edits, gleu_sentence_diacritic_insensitive

def test_gleu_perfect_is_one():
    assert gleu_sentence("on je otišao u školu", "on je otišao u školu") == 1.0

def test_gleu_worse_is_lower():
    good = gleu_sentence("on je otišao u školu", "on je otišao u školu")
    bad = gleu_sentence("on je otiša u skolu", "on je otišao u školu")
    assert bad < good

def test_prf_all_correct():
    gold = {("skolu", "školu"), ("otiša", "otišao")}
    pred = {("skolu", "školu"), ("otiša", "otišao")}
    p, r, f = prf_edits(pred, gold)
    assert (p, r) == (1.0, 1.0)
    assert abs(f - 1.0) < 1e-9

def test_prf_partial():
    gold = {("skolu", "školu"), ("otiša", "otišao")}
    pred = {("skolu", "školu")}
    p, r, f = prf_edits(pred, gold)
    assert p == 1.0
    assert r == 0.5

def test_diacritic_insensitive_ignores_kvacice():
    # model dodao kvacice, gold ih nema -> ne smije kazniti
    score = gleu_sentence_diacritic_insensitive("što volim", "sto volim")
    assert score == 1.0

def test_diacritic_insensitive_still_penalizes_real_errors():
    perfect = gleu_sentence_diacritic_insensitive("idem u skolu", "idem u skolu")
    bad = gleu_sentence_diacritic_insensitive("idem u skulo", "idem u skolu")
    assert bad < perfect
