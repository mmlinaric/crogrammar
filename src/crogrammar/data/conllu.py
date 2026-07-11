from dataclasses import dataclass, field


@dataclass(frozen=True)
class Token:
    form: str
    lemma: str
    upos: str
    feats: dict = field(default_factory=dict)


def _parse_feats(raw: str) -> dict:
    if raw == "_" or not raw:
        return {}
    out = {}
    for pair in raw.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
    return out


def parse_conllu(text: str):
    tokens = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if not line.strip():
            if tokens:
                yield tokens
                tokens = []
            continue
        cols = line.split("\t")
        if len(cols) < 6:
            continue
        if "-" in cols[0] or "." in cols[0]:
            continue
        tokens.append(Token(form=cols[1], lemma=cols[2], upos=cols[3], feats=_parse_feats(cols[5])))
    if tokens:
        yield tokens


def sentence_text(tokens) -> str:
    return " ".join(t.form for t in tokens)
