import pytest
import redactor
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
import re
from redactor import unredact

file = '/projects/unredact/docs/ex/12499_7.txt'
path = '/projects/unredact/docs/ex/*.txt'

def test_get_entity():
    with open(file, 'r') as afile:
        text = afile.read()
        vector, name = unredact.get_entity(text)
        assert len(vector[0]) == 5
        assert len(name) > 0

def test_doextraction():
    name, dic = unredact.doextraction(path)
    assert len(dic[0]) == 5

def test_get_entity_result():
    with open(file, 'r') as bfile:
        text = bfile.read()
        for sent in sent_tokenize(text):
            for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
                if(hasattr(chunk, 'label') and chunk.label() == 'PERSON'):
                    n = (' '.join(c[0] for c in chunk.leaves()))
                    for w in n:
                        tok = w.split()
                        for a in tok:
                            text = text.replace(a, len(a)*'\u2588', 1)
        vector, name = unredact.get_entity_result(text)
        reg = re.compile('\u2588+\s?\u2588*\s?\u2588*')
        for a in name:
            assert re.match(reg, a)

def test_doextraction_result():
    tentity, tvect = unredact.doextraction_result(path)
    assert tentity
    for eve in tvect:
        assert len(eve) == 5
