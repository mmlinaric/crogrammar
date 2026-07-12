def parse_raput(text: str):
    columns = None
    orig_idx = 10
    form_idx = 1
    forms = []
    origs = []
    misc = []
    for line in text.splitlines():
        if line.startswith("# global.columns"):
            names = line.split("=", 1)[1].split()
            columns = names
            if "RAPUT:ORIG" in names:
                orig_idx = names.index("RAPUT:ORIG")
            if "FORM" in names:
                form_idx = names.index("FORM")
            continue
        if line.startswith("#"):
            continue
        if not line.strip():
            if forms:
                yield _join(origs, forms, misc), _join_forms(forms, misc)
                forms, origs, misc = [], [], []
            continue
        cols = line.split("\t")
        if len(cols) <= orig_idx:
            continue
        if "-" in cols[0] or "." in cols[0]:
            continue
        form = cols[form_idx]
        orig = cols[orig_idx]
        if orig == "_" or not orig:
            orig = form
        forms.append(form)
        origs.append(orig)
        misc.append(cols[9] if len(cols) > 9 else "_")
    if forms:
        yield _join(origs, forms, misc), _join_forms(forms, misc)


def _join(origs, forms, misc):
    return _detokenize(origs, misc)


def _join_forms(forms, misc):
    return _detokenize(forms, misc)


def _detokenize(tokens, misc):
    out = ""
    for i, tok in enumerate(tokens):
        out += tok
        no_space = i < len(misc) and "SpaceAfter=No" in (misc[i] or "")
        if i < len(tokens) - 1 and not no_space:
            out += " "
    return out


def read_raput_pairs(path):
    with open(path, encoding="utf-8") as f:
        return list(parse_raput(f.read()))


def raput_training_pairs(path):
    pairs = []
    for src, tgt in read_raput_pairs(path):
        if src.lower() != tgt.lower():
            pairs.append({"src": src, "tgt": tgt})
    return pairs
