from crogrammar.inference.gec import CroatianGEC, Result


def dummy_fix(sentence: str) -> str:
    return sentence.replace("skolu", "školu").replace("otiša", "otišao")


def test_correct_returns_result():
    gec = CroatianGEC(generate_fn=dummy_fix)
    res = gec.correct("on je otiša u skolu")
    assert isinstance(res, Result)
    assert res.corrected == "on je otišao u školu"


def test_correct_produces_edits():
    gec = CroatianGEC(generate_fn=dummy_fix)
    res = gec.correct("on je otiša u skolu")
    kinds = {e["type"] for e in res.edits}
    assert "diacritic" in kinds
    assert "morphology" in kinds


def test_correct_multi_sentence():
    gec = CroatianGEC(generate_fn=dummy_fix)
    res = gec.correct("Idem u skolu. On je otiša.")
    assert "školu" in res.corrected
    assert "otišao" in res.corrected


def test_correct_clean_text_no_edits():
    gec = CroatianGEC(generate_fn=lambda s: s)
    res = gec.correct("Sve je u redu.")
    assert res.edits == []
    assert res.corrected == "Sve je u redu."


def test_result_to_dict():
    gec = CroatianGEC(generate_fn=dummy_fix)
    res = gec.correct("skolu")
    d = res.to_dict()
    assert set(d.keys()) == {"corrected", "edits"}
