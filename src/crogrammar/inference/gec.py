from dataclasses import dataclass, field

from .segment import split_sentences
from .diff import compute_edits


@dataclass
class Result:
    corrected: str
    edits: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"corrected": self.corrected, "edits": self.edits}


class CroatianGEC:
    def __init__(self, model_path: str = None, generate_fn=None, max_len: int = 256):
        self.max_len = max_len
        if generate_fn is not None:
            self._generate = generate_fn
        elif model_path is not None:
            self._generate = self._build_hf_generate(model_path)
        else:
            raise ValueError("Zadaj model_path ili generate_fn")

    def _build_hf_generate(self, model_path):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()

        def _gen(sentence: str) -> str:
            inputs = tok("ispravi: " + sentence, return_tensors="pt",
                         truncation=True, max_length=self.max_len)
            out = model.generate(**inputs, max_length=self.max_len)
            return tok.decode(out[0], skip_special_tokens=True)

        return _gen

    def correct(self, text: str) -> Result:
        sentences = split_sentences(text)
        corrected_sentences = [self._generate(s) for s in sentences]
        corrected = " ".join(corrected_sentences)
        edits = compute_edits(text, corrected) if sentences else []
        return Result(corrected=corrected, edits=edits)
