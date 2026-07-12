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
