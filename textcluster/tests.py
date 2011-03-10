from nose.tools import eq_
from textcluster import Corpus

docs = (
        'Every good boy does fine.',
        'Every good girl does well.',
        'Cats eat rats.',
        "Rats don't sleep.",
        )


def test_cluster():
    c = Corpus(similarity=0.1)
    for doc in docs:
        c.add(doc)

    groups = c.cluster()

    eq_(len(groups), 2)
    eq_(len(groups[0].similars), 1)
    eq_(len(groups[1].similars), 1)
