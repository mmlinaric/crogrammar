from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    base_model: str = "google/byt5-small"
    task_prefix: str = "ispravi: "
    train_file: str = "data/processed/train.jsonl"
    dev_file: str = "data/processed/dev.jsonl"
    output_dir: str = "models/byt5-hr-gec"
    max_source_len: int = 256
    max_target_len: int = 256
    batch_size: int = 4
    grad_accum: int = 8
    num_epochs: int = 3
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    save_steps: int = 1000
    eval_steps: int = 1000
    seed: int = 42
    fp16: bool = False

    def as_dict(self) -> dict:
        return asdict(self)
