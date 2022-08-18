import unittest
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_adaptive import CorpusGraph
from pyterrier_pisa import PisaIndex
import tempfile

class TestCorpusGraph(unittest.TestCase):
    def test_vaswani_from_retriever(self):
        dataset = pt.get_dataset('irds:vaswani')
        with tempfile.TemporaryDirectory() as dout:
            indexer = PisaIndex(f'{dout}/index.pisa')
            indexer.index(dataset.get_corpus_iter())
            bm25 = PisaIndex(f'{dout}/index.pisa').bm25(num_results=5)
            graph = CorpusGraph.from_retriever(bm25, dataset.get_corpus_iter(), f'{dout}/graph', k=4)

if __name__ == '__main__':
    unittest.main()
