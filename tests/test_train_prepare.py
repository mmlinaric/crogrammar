import json
from crogrammar.train.train import preprocess_batch

class FakeTok:
    pad_token_id = 0
    def __call__(self, texts, max_length=None, truncation=True, padding=False):
        return {"input_ids": [[len(t)] for t in texts],
                "attention_mask": [[1] for _ in texts]}

def test_preprocess_batch_adds_prefix_and_labels():
    batch = {"src": ["skolu", "otiša"], "tgt": ["školu", "otišao"]}
    tok = FakeTok()
    out = preprocess_batch(batch, tok, prefix="ispravi: ", max_src=32, max_tgt=32)
    assert "input_ids" in out
    assert "labels" in out
    assert len(out["input_ids"]) == 2
