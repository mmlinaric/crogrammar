from crogrammar.train.config import TrainConfig

def test_defaults():
    cfg = TrainConfig()
    assert cfg.base_model == "google/byt5-small"
    assert cfg.task_prefix == "ispravi: "
    assert cfg.max_source_len > 0
    assert cfg.seed == 42

def test_override():
    cfg = TrainConfig(num_epochs=5, batch_size=8)
    assert cfg.num_epochs == 5
    assert cfg.batch_size == 8

def test_as_dict_serializable():
    import json
    cfg = TrainConfig()
    json.dumps(cfg.as_dict())  # ne smije baciti
