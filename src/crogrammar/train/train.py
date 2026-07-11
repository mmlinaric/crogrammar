def preprocess_batch(batch, tokenizer, prefix, max_src, max_tgt):
    sources = [prefix + s for s in batch["src"]]
    model_inputs = tokenizer(sources, max_length=max_src, truncation=True)
    labels = tokenizer(batch["tgt"], max_length=max_tgt, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train(cfg, resume_from_checkpoint=None):
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model)

    data = load_dataset("json", data_files={"train": cfg.train_file, "dev": cfg.dev_file})
    tokenized = data.map(
        lambda b: preprocess_batch(b, tokenizer, cfg.task_prefix,
                                   cfg.max_source_len, cfg.max_target_len),
        batched=True, remove_columns=data["train"].column_names,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.num_epochs,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=cfg.fp16,
        bf16=getattr(cfg, "bf16", False),
        seed=cfg.seed,
        logging_steps=100,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=tokenized["train"], eval_dataset=tokenized["dev"],
        data_collator=collator, processing_class=tokenizer,
    )
    if resume_from_checkpoint is True:
        from transformers.trainer_utils import get_last_checkpoint
        import os
        resume_from_checkpoint = (
            get_last_checkpoint(cfg.output_dir)
            if os.path.isdir(cfg.output_dir) else None
        )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir
