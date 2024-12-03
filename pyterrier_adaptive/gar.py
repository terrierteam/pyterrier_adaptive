from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import pyterrier_adaptive
logger = ir_datasets.log.easy()


class GAR(pt.Transformer):
    """A :class:`~pyterrier.Transformer` that implements Graph-based Adaptive Re-ranking (GAR).

    Required input columns: ``['qid', 'query', 'docno', 'score', 'rank']``

    Output columns: ``['qid', 'query', 'docno', 'score', 'rank', 'iteration']``

    .. note::

        The iteration column defines the batch number that first identified the document in the
        results. Due to the alternating nature of the algorithm, ``even=initial retrieval``, ``odd=corpus graph``,
        and ``-1=backfilled``.
    
    .. cite.dblp:: conf/cikm/MacAvaneyTM22
    """
    def __init__(self,
        scorer: pt.Transformer,
        corpus_graph: 'pyterrier_adaptive.CorpusGraph',
        num_results: int = 1000,
        batch_size: Optional[int] = None,
        backfill: bool = True,
        enabled: bool = True,
        verbose: bool = False):
        """
        Args:
            scorer(:class:`~pyterrier.Transformer`): A transformer that scores query-document pairs. It will only be provided with ['qid, 'query', 'docno', 'score'].
            corpus_graph(:class:`~pyterrier_adaptive.CorpusGraph`): A graph of the corpus, enabling quick lookups of nearest neighbours
            num_results(int): The maximum number of documents to score (called "budget" and $c$ in the paper)
            batch_size(int): The number of documents to score at once (called $b$ in the paper). If not provided, will attempt to use the batch size from the scorer
            backfill(bool): If True, always include all documents from the initial stage, even if they were not re-scored
            enabled(bool): If False, perform re-ranking without using the corpus graph
            verbose(bool): If True, print progress information
        """
        self.scorer = scorer
        self.corpus_graph = corpus_graph
        self.num_results = num_results
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(scorer, 'batch_size') else 16
        self.batch_size = batch_size
        self.backfill = backfill
        self.enabled = enabled
        self.verbose = verbose

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Graph-based Adaptive Re-ranking to the provided dataframe. Essentially,
        Algorithm 1 from the paper.
        """
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()
        if self.verbose:
            qids = logger.pbar(qids, desc='adaptive re-ranking', unit='query')

        for qid in qids:
            query = df[qid]['query'].iloc[0]
            scores = {}
            res_map = [Counter(dict(zip(df[qid].docno, df[qid].score)))] # initial results
            if self.enabled:
                res_map.append(Counter()) # frontier
            frontier_data = {'minscore': float('inf')}
            iteration = 0
            while len(scores) < self.num_results and any(r for r in res_map):
                if len(res_map[iteration%len(res_map)]) == 0:
                    # if there's nothing available for the one we select, skip this iteration (i.e., move on to the next one)
                    iteration += 1
                    continue
                this_res = res_map[iteration%len(res_map)] # alternate between the initial ranking and frontier
                size = min(self.batch_size, self.num_results - len(scores)) # get either the batch size or remaining budget (whichever is smaller)
                
                # build batch of documents to score in this round
                batch = this_res.most_common(size)
                batch = pd.DataFrame(batch, columns=['docno', 'score'])
                batch['qid'] = qid
                batch['query'] = query

                # go score the batch of document with the re-ranker
                batch = self.scorer(batch)
                scores.update({k: (s, iteration) for k, s in zip(batch.docno, batch.score)})
                self._drop_docnos_from_counters(batch.docno, res_map)
                if len(scores) < self.num_results and self.enabled:
                    self._update_frontier(batch, res_map[1], frontier_data, scores)
                iteration += 1

            # Add scored items to results
            result['qid'].append(np.full(len(scores), qid))
            result['query'].append(np.full(len(scores), query))
            result['rank'].append(np.arange(len(scores)))
            for did, (score, i) in Counter(scores).most_common():
                result['docno'].append(did)
                result['score'].append(score)
                result['iteration'].append(i)

            # Backfill unscored items
            if self.backfill and len(scores) < self.num_results:
                last_score = result['score'][-1] if result['score'] else 0.
                count = min(self.num_results - len(scores), len(res_map[0]))
                result['qid'].append(np.full(count, qid))
                result['query'].append(np.full(count, query))
                result['rank'].append(np.arange(len(scores), len(scores) + count))
                for i, (did, score) in enumerate(res_map[0].most_common()):
                    if i >= count:
                        break
                    result['docno'].append(did)
                    result['score'].append(last_score - 1 - i)
                    result['iteration'].append(-1)

        return pd.DataFrame({
            'qid': np.concatenate(result['qid']),
            'query': np.concatenate(result['query']),
            'docno': result['docno'],
            'rank': np.concatenate(result['rank']),
            'score': result['score'],
            'iteration': result['iteration'],
        })

    def _update_frontier(self, scored_batch, frontier, frontier_data, scored_dids):
        remaining_budget = self.num_results - len(scored_dids)
        for score, did in sorted(zip(scored_batch.score, scored_batch.docno), reverse=True):
            if len(frontier) < remaining_budget or score >= frontier_data['minscore']:
                hit = False
                for target_did in self.corpus_graph.neighbours(did):
                    if target_did not in scored_dids:
                        if target_did not in frontier or score > frontier[target_did]:
                            frontier[target_did] = score
                            hit = True
                if hit and score < frontier_data['minscore']:
                    frontier_data['minscore'] = score

    def _drop_docnos_from_counters(self, docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]
