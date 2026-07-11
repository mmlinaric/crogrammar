from crogrammar.eval.metrics import gleu_sentence, prf_edits

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
