"""
Takes a json dump of sumo data and clusters it together using mathematics.
"""
import math
import csv
import os
import sys

from stemming.porter2 import stem
from numpy import sqrt, dot


SIM_THRESHOLD = 0.3


def tokenize(str):
    # lowercase
    return [stem(c.strip("""\\.!?,(){}[]"'""")) for c in str.lower().split()]



class Document():
    tf = {}
    similar = []
    _idf = None
    _vector = None

    def __init__(self, corpus, document):
        self.corpus = corpus
        self.document = document
        self.words = tokenize(document)
        # for word in set(self.words):
        #    self.tf[word] = self.words.count(word) / float(len(self.words))

    def __unicode__(self):
        return self.document

    def idf(self, cached=True):
        if cached and self._idf:
            return self._idf

        num_docs = len(self.corpus.docs)
        idf = {}
        for word in self.tf.keys():
            num_occurences = len([d for d in self.corpus.docs.values()
                                  if d.tf.get(word) is not None])

            idf[word] = math.log(num_docs / (1.0 + num_occurences))
        self._idf = idf
        return idf

    def tf_idf(self):
        tf_idf = {}
        idf = self.idf()
        for word in self.tf.keys():
            tf_idf[word] = idf[word] * self.tf[word]
        return tf_idf

    def vector(self, cached=True):
        if self._vector and cached:
            return self._vector

        v = []
        tf_idf = self.tf_idf()
        for word in self.corpus.words.keys():
            v.append(tf_idf.get(word))

        self._vector = v
        return v


class Corpus():
    """Document corpus which calculates Term Frequency/Inverse Document
    Frequency."""

    def __init__(self):
        self.docs = {}
        self.words = {}

    def load(self, key, document):
        """Adds a document to the corpus."""
        doc = Document(self, document)
        for k in doc.tf.keys():
            self.words[k] = 1
        self.docs[key] = doc

mag = lambda x: sqrt(dot(x, x))

def tanimoto(doc, other):
    """
    We can pull the tf.idf dictionaries and turn them into vectors.
    Tanimoto is dot(A,B)/(magnitude(A)^2+magnitude(B)^2 - AB) where
    A and B are vectors.
    """
    v1 = doc.vector()
    v2 = other.vector()
    dp = dot(v1, v2)
    return dp / (mag(v1)**2 + mag(v2)**2 - dp)

def tanimoto(doc, other):
    v1 = set(doc.words)
    v2 = other.words
    i = v1.intersection(v2)
    return float(len(i))/(len(v1)+len(v2)-len(i))


def cluster(corpus):
    c = {}
    seen = {}
    i = 0
    for key in corpus.docs.keys():
        i += 1
        if i % 10000 == 0:
            print i
        if seen.get(key):
            continue
        seen[key] = 1
        similars = [] # key of similar items
        for c_key in corpus.docs.keys():
            if seen.get(c_key) is not None:
                continue
            if tanimoto(corpus.docs[key], corpus.docs[c_key]) > SIM_THRESHOLD:
                similars.append(c_key)
                seen[c_key] = 1

        c[key] = similars
    return c



def group_support_issues():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = os.path.join(os.path.dirname(__file__), 'data/result.csv')

    reader = csv.reader(open(filename))

    corpus = Corpus()
    print "Loading Data"
    i = 1
    for row in reader:
        key = int(row[0])
        msg = row[2]
        if i % 10000 == 0:
            print i
        corpus.load(key, msg)
        i += 1

    print "Clustering Data"
    for doc, friends in cluster(corpus).iteritems():
        if len(friends) == 0:
            continue
        print "* " + corpus.docs[doc].document
        for friend in friends:
            print "   *  " + corpus.docs[friend].document

if __name__ == "__main__":
    group_support_issues()
