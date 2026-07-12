from crogrammar.data.raput import parse_raput

SAMPLE = """# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC RAPUT:ORIG RAPUT:ERRORS
# sent_id = 1
# text = akcijske filmove.
1\takcijske\takcijski\tADJ\t_\t_\t_\t_\t_\t_\takciske\tPHON-SEG
2\tfilmove\tfilm\tNOUN\t_\t_\t_\t_\t_\tSpaceAfter=No\tfilmove\t_
3\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_\t.\t_

# sent_id = 2
# text = moja kuća
1\tmoja\tmoj\tDET\t_\t_\t_\t_\t_\t_\tMoja\t_
2\tkuća\tkuća\tNOUN\t_\t_\t_\t_\t_\t_\tkuca\tORTHO
"""


def test_parse_raput_yields_pairs():
    pairs = list(parse_raput(SAMPLE))
    assert len(pairs) == 2


def test_src_is_orig_and_tgt_is_form():
    pairs = list(parse_raput(SAMPLE))
    src, tgt = pairs[0]
    assert "akciske" in src        # ORIG (napisano)
    assert "akcijske" in tgt       # FORM (ispravak)


def test_second_pair_diacritic_correction():
    pairs = list(parse_raput(SAMPLE))
    src, tgt = pairs[1]
    assert src == "Moja kuca"
    assert tgt == "moja kuća"


def test_falls_back_to_form_when_orig_underscore():
    text = (
        "# sent_id = 3\n"
        "1\tmore\tmore\tNOUN\t_\t_\t_\t_\t_\t_\t_\t_\n"
    )
    pairs = list(parse_raput(text))
    assert pairs == [("more", "more")]


def test_raput_training_pairs_extracts_real_errors(tmp_path):
    from crogrammar.data.raput import raput_training_pairs
    sample = (
        "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC RAPUT:ORIG RAPUT:ERRORS\n"
        "# sent_id = 1\n"
        "1\takcijske\takcijski\tADJ\t_\t_\t_\t_\t_\t_\takciske\tPHON-SEG\n"
        "2\tfilm\tfilm\tNOUN\t_\t_\t_\t_\t_\t_\tfilm\t_\n"
        "\n"
        "# sent_id = 2\n"
        "1\tmoja\tmoj\tDET\t_\t_\t_\t_\t_\t_\tMoja\t_\n"
    )
    p = tmp_path / "r.conllup"
    p.write_text(sample, encoding="utf-8")
    pairs = raput_training_pairs(str(p))
    # sent 1 ima pravu gresku (akciske->akcijske) => ukljucen
    assert any(x["src"] == "akciske film" and x["tgt"] == "akcijske film" for x in pairs)
    # sent 2 je samo case (Moja->moja) => iskljucen
    assert not any(x["src"] == "Moja" for x in pairs)


def test_raput_training_pairs_returns_dicts(tmp_path):
    from crogrammar.data.raput import raput_training_pairs
    sample = (
        "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC RAPUT:ORIG RAPUT:ERRORS\n"
        "# sent_id = 1\n"
        "1\tdoći\tdoći\tVERB\t_\t_\t_\t_\t_\t_\tdoć\tORTHO\n"
    )
    p = tmp_path / "r.conllup"
    p.write_text(sample, encoding="utf-8")
    pairs = raput_training_pairs(str(p))
    assert pairs == [{"src": "doć", "tgt": "doći"}]
