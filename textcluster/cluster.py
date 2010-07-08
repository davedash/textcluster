"""
Takes a json dump of sumo data and clusters it together using mathematics.
"""
import math
import csv
import os
import sys
from collections import defaultdict

from stemming.porter2 import stem

from search import STOPWORDS

SIM_THRESHOLD = .1
MIN_DOCUMENT_LENGTH = 3

def tokenize(str):
    # lowercase
    strips = """\\.!?,(){}[]"'"""
    return [stem(c.strip(strips)) for c in str.lower().split()
            if STOPWORDS.get(c.strip(strips)) is None]


class Document():

    def __init__(self, corpus, obj):
        self.corpus = corpus
        self.object = obj
        self.document = unicode(obj)
        self.tf = {}
        self._tf_idf = None
        words = tokenize(self.document)
        for word in set(words):
            self.tf[word] = words.count(word) / float(len(words))

    def __repr__(self):
        return self.document

    def idf(self, cached=True):

        num_docs = len(self.corpus.docs)
        idf = {}
        for word in self.tf.keys():
            num_occurences = self.corpus.words.get(word, 0)
            idf[word] = math.log(num_docs / (1.0 + num_occurences))

        return idf

    def tf_idf(self, cached=True):
        if self._tf_idf and cached:
            return self._tf_idf

        self._tf_idf = {}
        idf = self.idf()
        for word in self.tf.keys():
            self._tf_idf[word] = idf[word] * self.tf[word]

        return self._tf_idf


class Corpus():
    """Document corpus which calculates Term Frequency/Inverse Document
    Frequency."""

    def __init__(self, similarity=SIM_THRESHOLD):
        self.similarity = similarity
        self.docs = {}
        self.words = defaultdict(int)
        self.index = defaultdict(dict)

    def add(self, document, key=None):
        """Adds a document to the corpus."""
        if not key:
            try:
                key = document.id
            except AttributeError:
                key = document

        doc = Document(self, document)

        if len(doc.tf) < MIN_DOCUMENT_LENGTH:
            return

        for k in doc.tf.keys():
            if k in self.words:
                self.words[k] += 1

        self.docs[key] = doc

    def create_index(self):
        index = {}
        for id, doc in self.docs.iteritems():
            for word, weight in doc.tf_idf().iteritems():
                self.index[word][id] = weight


    def cluster(self):
        seen = {}
        scores = {}
        self.create_index()
        for key, doc in self.docs.iteritems():
            if seen.get(key):
                continue
            seen[key] = 1
            scores[key] = defaultdict(int)

            for word, o_weight in doc.tf_idf().iteritems():
                if word in self.index:
                    matches = self.index[word]

                    for c_key, c_weight in matches.iteritems():
                        if seen.get(c_key):
                            continue
                        scores[key][c_key] += o_weight * c_weight

            scores[key] = dict(((k, v) for k, v in scores[key].iteritems()
                               if v >= self.similarity))
            seen.update(scores[key])

        scores = sorted(scores.iteritems(),
                        cmp=lambda x, y: cmp(len(x[1]), len(y[1])),
                        reverse=True)

        groups = []
        for score in scores:
            g = Group()
            g.primary = self.docs[score[0]]
            for id, similarity in score[1].iteritems():
                g.add_similar(self.docs[id].object, similarity)
            groups.append(g)

        return groups

class Group:
    primary = None
    similars = []

    def add_similar(self, obj, similarity):
        self.similars.append(dict(object=obj, similarity=similarity))

