from crogrammar.eval.run_eval import evaluate_pairs, evaluate_pairs_batched


def test_evaluate_pairs_perfect():
    pairs = [{"src": "skolu", "tgt": "školu"}]
    fix = lambda s: "školu"
    score = evaluate_pairs(pairs, fix)
    assert score["gleu"] == 1.0


def test_evaluate_pairs_reports_count():
    pairs = [{"src": "a", "tgt": "a"}, {"src": "b", "tgt": "b"}]
    score = evaluate_pairs(pairs, lambda s: s)
    assert score["n"] == 2


def test_evaluate_pairs_batched_matches_count():
    pairs = [{"src": str(i), "tgt": str(i)} for i in range(70)]
    score = evaluate_pairs_batched(pairs, lambda batch: list(batch), batch_size=32)
    assert score["n"] == 70
    assert score["gleu"] == 1.0
