from dataclasses import dataclass, field

from .segment import split_sentences
from .diff import compute_edits


@dataclass
class Result:
    corrected: str
    edits: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"corrected": self.corrected, "edits": self.edits}


def _select_device(prefer: str = "auto"):
    import torch
    if prefer not in ("auto", "cuda", "directml", "cpu"):
        prefer = "auto"
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if prefer in ("auto", "directml"):
        try:
            import torch_directml as dml
            if dml.is_available():
                return dml.device(), "directml"
        except Exception:
            pass
    return torch.device("cpu"), "cpu"


class CroatianGEC:
    def __init__(self, model_path: str = None, generate_fn=None, max_len: int = 256,
                 device: str = "auto"):
        self.max_len = max_len
        self.device_kind = None
        if generate_fn is not None:
            self._generate = generate_fn
        elif model_path is not None:
            self._generate = self._build_hf_generate(model_path, device)
        else:
            raise ValueError("Zadaj model_path ili generate_fn")

    def _build_hf_generate(self, model_path, prefer_device="auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()

        device, kind = _select_device(prefer_device)
        cpu_device = torch.device("cpu")
        try:
            model.to(device)
        except Exception:
            device, kind = cpu_device, "cpu"
            model.to(device)
        self.device_kind = kind
        self._model = model
        self._tok = tok

        def _run(texts, target_device):
            inputs = tok(["ispravi: " + s for s in texts], return_tensors="pt",
                         truncation=True, max_length=self.max_len,
                         padding=True).to(target_device)
            with torch.no_grad():
                out = model.generate(**inputs, max_length=self.max_len)
            return tok.batch_decode(out, skip_special_tokens=True)

        def _gen_batch(sentences):
            if not sentences:
                return []
            try:
                return _run(sentences, device)
            except Exception as e:
                if kind != "cpu":
                    model.to(cpu_device)
                    self.device_kind = "cpu (fallback)"
                    return _run(sentences, cpu_device)
                raise e

        def _gen(sentence: str) -> str:
            return _gen_batch([sentence])[0]

        self.generate_batch = _gen_batch
        return _gen

    def correct(self, text: str) -> Result:
        sentences = split_sentences(text)
        corrected_sentences = [self._generate(s) for s in sentences]
        corrected = " ".join(corrected_sentences)
        edits = compute_edits(text, corrected) if sentences else []
        return Result(corrected=corrected, edits=edits)
