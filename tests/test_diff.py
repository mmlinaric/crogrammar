from crogrammar.inference.diff import compute_edits, classify_edit

def test_no_edits_when_identical():
    assert compute_edits("On je tu", "On je tu") == []

def test_single_word_replacement():
    edits = compute_edits("Idem u skolu", "Idem u školu")
    assert len(edits) == 1
    e = edits[0]
    assert e["original"] == "skolu"
    assert e["suggestion"] == "školu"
    assert "Idem u skolu"[e["start"]:e["end"]] == "skolu"

def test_classify_diacritic():
    assert classify_edit("skola", "škola") == "diacritic"

def test_classify_morphology():
    assert classify_edit("otiša", "otišao") == "morphology"

def test_classify_spelling_default():
    assert classify_edit("teik", "tekst") == "spelling"

def test_multiple_edits_positions():
    edits = compute_edits("on je otiša u skolu", "on je otišao u školu")
    originals = [e["original"] for e in edits]
    assert originals == ["otiša", "skolu"]
    for e in edits:
        assert "on je otiša u skolu"[e["start"]:e["end"]] == e["original"]
