from crogrammar.eval.run_eval import evaluate_pairs


def test_evaluate_pairs_perfect():
    pairs = [{"src": "skolu", "tgt": "školu"}]
    fix = lambda s: "školu"
    score = evaluate_pairs(pairs, fix)
    assert score["gleu"] == 1.0


def test_evaluate_pairs_reports_count():
    pairs = [{"src": "a", "tgt": "a"}, {"src": "b", "tgt": "b"}]
    score = evaluate_pairs(pairs, lambda s: s)
    assert score["n"] == 2
