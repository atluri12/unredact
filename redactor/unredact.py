import glob
import io
import os
import pdb
import sys
import nltk
import re
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from nltk.util import ngrams
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def get_entity(text):
    """Prints the entity inside of the text."""
    entities, tokens, token_length, length, first, second, vector = [], [], [], [], [], [], []
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                name = ' '.join(c[0] for c in chunk.leaves())
                entities.append(name)
                token_length.append(len(name))
                words = nltk.word_tokenize(name)
                tokens.append(len(words))
                length.append(name.count(' '))
                first.append(len(words[0]))
                if(len(words) > 1):
                    second.append(len(words[1]))
                else:
                    second.append(0)
    for tok_len, tok, leng, fir, sec in zip(token_length, tokens, length, first, second):
        vector.append({'size': tok_len, 'wordcount': tok, 'length': leng, 'firstword': fir, 'secondword': sec})
    return (vector, entities)


def doextraction(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    vector, entities = [], []
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            vec, name = get_entity(text)
            vector.extend(vec)
            entities.extend(name)
    return (entities, vector)

def get_entity_result(text):
    tokens, token_length, length, first, second, vector = [], [], [], [], [], []
    exp = re.compile('\u2588+\s?\u2588*\s?\u2588*')
    entities = re.findall(exp, text)
    for name in entities:
        token_length.append(len(name))
        length.append(name.count(' '))
        words = name.split(' ')
        tokens.append(len(words))
        first.append(len(words[0]))
        if(len(words) > 1):
            second.append(len(words[1]))
        else:
            second.append(0)
    for tok_len, tok, leng, fir, sec in zip(token_length, tokens, length, first, second):
        vector.append({'size': tok_len, 'wordcount': tok, 'length': leng, 'firstword': fir, 'secondword': sec})
    return (vector, entities)

def doextraction_result(glob_text):
    entities, vector = [], []
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            token = []
            for sent in sent_tokenize(text):
                for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
                    if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                        entities.append(' '.join(c[0] for c in chunk.leaves()))
                        token.append(' '.join(c[0] for c in chunk.leaves()))
            for word in token:
                tok = word.split()
                for a in tok:
                    block = len(a)
                    text = text.replace(a, '\u2588'*block, 1)    
            vec, name = get_entity_result(text)
            vector.extend(vec)
    return (entities, vector)

if __name__ == '__main__':
    train_entities, train_vector = doextraction('docs/train/*.txt')
    train_dictvec = DictVectorizer()
    train = train_dictvec.fit_transform(train_vector).toarray()
    print(train)
    classifier = GaussianNB()
    classifier.fit(train, train_entities)
    v = KNeighborsClassifier(n_neighbors = 3)
    vneigh = v.fit(train, train_entities)
    test_entities, test_vector = doextraction_result('docs/test/*.txt')
    test_dictvec = DictVectorizer()
    test = test_dictvec.fit_transform(test_vector).toarray()
    result = classifier.predict(test)
    values = vneigh.kneighbors(test, n_neighbors=3, return_distance=False)
    if(len(test_entities) < len(result)):
        a = precision_recall_fscore_support(test_entities, result[:len(test_entities)], average = 'micro')
    else:
        a = precision_recall_fscore_support(test_entities[:len(result)], result, average = 'micro')
    with open('output.txt', 'w') as f:
        for ev, num in zip(values, test_entities):
            f.write('Word is: ' + num + '\n')
            for one in ev:
                f.write(train_entities[one] + '\n')
            f.write('\n')
