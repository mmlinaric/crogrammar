from crogrammar.data.conllu import parse_conllu, Token

SAMPLE = """# sent_id = 1
# text = On ide.
1\tOn\ton\tPRON\tPp3msn\tCase=Nom|Gender=Masc\t_\t_
2\tide\tici\tVERB\tVmr3s\tPerson=3|Number=Sing\t_\t_
3\t.\t.\tPUNCT\tZ\t_\t_\t_

# sent_id = 2
# text = Ja spavam.
1\tJa\tja\tPRON\tPp1-sn\tCase=Nom\t_\t_
2\tspavam\tspavati\tVERB\tVmr1s\tPerson=1|Number=Sing\t_\t_
3\t.\t.\tPUNCT\tZ\t_\t_\t_
"""

def test_parse_conllu_returns_sentences():
    sentences = list(parse_conllu(SAMPLE))
    assert len(sentences) == 2

def test_sentence_has_tokens_with_features():
    sentences = list(parse_conllu(SAMPLE))
    first = sentences[0]
    assert first[0] == Token(form="On", lemma="on", upos="PRON", feats={"Case": "Nom", "Gender": "Masc"})
    assert first[1].form == "ide"

def test_reconstruct_text_joins_forms():
    from crogrammar.data.conllu import sentence_text
    sentences = list(parse_conllu(SAMPLE))
    assert sentence_text(sentences[0]) == "On ide ."
