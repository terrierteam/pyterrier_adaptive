Adaptive Retrieval
=========================================================

`pyterrier-adaptive <https://github.com/terrierteam/pyterrier_adaptive>`__ provides PyTerrier
functionality to support Adaptive Retrieval.

Adaptive Retrieval is a family of techniques that help overcome the recall limitation of
re-ranking approaches by identifying relevant documents that were missed by earlier stages.

API Documentation
---------------------------------------------------------

.. autoclass:: pyterrier_adaptive.GAR
    :members:

.. autoclass:: pyterrier_adaptive.CorpusGraph
    :members:

.. autoclass:: pyterrier_adaptive.NpTopKCorpusGraph
    :members:

.. autoclass:: pyterrier_adaptive.Laff
    :members:

Bibliography
---------------------------------------------------------

For more information on adaptive retrieval, see:

.. cite.dblp:: conf/cikm/MacAvaneyTM22
.. cite.dblp:: conf/cikm/MacAvaneyTM22a
.. cite.dblp:: conf/sigir/KulkarniMGF23
.. cite.dblp:: conf/ecir/FraylingMMO24
.. cite.dblp:: conf/sigir/MacAvaneyT24
.. cite.dblp:: journals/corr/abs-2410-20286
