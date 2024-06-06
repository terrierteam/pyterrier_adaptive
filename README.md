# pyterrier_adaptive

[PyTerrier](http://github.com/terrier-org/pyterrier) implementation of [Adaptive Re-Ranking using a Corpus Graph](https://arxiv.org/abs/2208.08942) (CIKM 2022).

## Getting Started

Install with pip:

```bash
pip install --upgrade git+https://github.com/terrierteam/pyterrier_adaptive.git
```

Basic Example over the MS MARCO passage corpus (making use of the [pyterrier_t5](https://github.com/terrierteam/pyterrier_t5) and [pyterrier_pisa](https://github.com/terrierteam/pyterrier_pisa) plugins):

Try examples in Google Colab! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrierteam/pyterrier_adaptive/blob/master/examples/example.ipynb)

```python
import pyterrier as pt
pt.init()
from pyterrier_t5 import MonoT5ReRanker
from pyterrier_pisa import PisaIndex
from pyterrier_adaptive import GAR, CorpusGraph

dataset = pt.get_dataset('irds:msmarco-passage')
retriever = PisaIndex.from_dataset('msmarco_passage').bm25()
scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=16)
graph = CorpusGraph.from_dataset('msmarco_passage', 'corpusgraph_bm25_k16').to_limit_k(8)

pipeline = retriever >> GAR(scorer, graph) >> pt.text.get_text(dataset, 'text')

pipeline.search('clustering hypothesis information retrieval')
# qid                                        query    docno  rank       score  iteration                                               text
#   1  clustering hypothesis information retrieval  2180710     0   -0.017059          0  Cluster analysis or clustering is the task of ...
#   1  clustering hypothesis information retrieval  8430269     1   -0.166563          1  Clustering is the grouping of a particular set...
#   1  clustering hypothesis information retrieval  1091429     2   -0.208345          1  Clustering is a fundamental data analysis meth...
#   1  clustering hypothesis information retrieval  2180711     3   -0.341018          5  Cluster analysis or clustering is the task of ...
#   1  clustering hypothesis information retrieval  6031959     4   -0.367014          5  Cluster analysis or clustering is the task of ...
#  ..                                          ...      ...   ...         ...        ...                                                ...
#                iteration column indicates which GAR batch the document was scored in ^
#                even=initial retrieval   odd=corpus graph    -1=backfilled
```

Evaluation on a test collection ([TREC DL19](https://ir-datasets.com/msmarco-passage#msmarco-passage/trec-dl-2019)):

```python
from pyterrier.measures import *
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
pt.Experiment(
    [retriever, retriever >> scorer, retriever >> GAR(scorer, graph)],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG, MAP(rel=2), R(rel=2)@1000],
    names=['bm25', 'bm25 >> monot5', 'bm25 >> GAR(monot5)']
)
#                name      nDCG  AP(rel=2)  R(rel=2)@1000
#                bm25  0.602325   0.303099       0.755495
#      bm25 >> monot5  0.696293   0.481259       0.755495
# bm25 >> GAR(monot5)  0.724501   0.489978       0.825952
```

## Reproduction

Detailed instructions to come!

## Building a Corpus Graph

You can construct a $k$ corpus graph using any retriever transformer and a corpus iterator.

Example:

```python
from pyterrier_adaptive import CorpusGraph
from pyterrier_pisa import PisaIndex
dataset = pt.get_dataset('irds:msmarco-passage')

# Build the index needed for BM25 retrieval (if it doesn't already exist)
idx = PisaIndex('msmarco-passage.pisa', threads=45) # adjust for your resources
if not idx.built():
    idx.index(dataset.get_corpus_iter())

# Build the corpus graph
K = 16 # number of nearest neighbours
graph16 = CorpusGraph.from_retriever(
    idx.bm25(num_results=K+1), # K+1 needed because retriever will return original document
    dataset.get_corpus_iter(),
    'msmarco-passage.gbm25.16',
    k=K)
```

You can load a corpus graph using the `.load(path)` function. You can simulate lower $k$ values
using `.to_limit_k(k)`

```python
graph16 = CorpusGraph.load('msmarco-passage.gbm25.16')
graph8 = graph16.to_limit_k(8)
```

# Citation

Adaptive Re-Ranking with a Corpus Graph. Sean MacAvaney, Nicola Tonellotto and Craig Macdonald. In Proceedings of CIKM 2022.

```bibtex
@inproceedings{gar2022,
  title = {Adaptive Re-Ranking with a Corpus Graph},
  booktitle = {Proceedings of ACM CIKM},
  author = {Sean MacAvaney and Nicola Tonellotto and Craig Macdonald},
  year = 2022
}
```
