from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    base_model: str = "google/byt5-small"
    task_prefix: str = "ispravi: "
    train_file: str = "data/processed/train.jsonl"
    dev_file: str = "data/processed/dev.jsonl"
    output_dir: str = "models/byt5-hr-gec"
    max_source_len: int = 128
    max_target_len: int = 128
    batch_size: int = 16
    grad_accum: int = 2
    num_epochs: int = 1
    learning_rate: float = 5e-4
    warmup_steps: int = 300
    save_steps: int = 500
    eval_steps: int = 500
    seed: int = 42
    fp16: bool = False
    bf16: bool = False

    def as_dict(self) -> dict:
        return asdict(self)
