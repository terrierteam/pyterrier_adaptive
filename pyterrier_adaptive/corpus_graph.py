import pickle
import tempfile
from lz4.frame import LZ4FrameFile
import json
import numpy as np
import pandas as pd
from pathlib import Path
import more_itertools
from typing import Union, Tuple, List
import ir_datasets
from npids import Lookup
import pyterrier_alpha as pta

try:
  from functools import cached_property
except ImportError:
  # polyfill for python<3.8, adapted from <https://github.com/pydanny/cached-property>
  class cached_property(object):
    def __init__(self, fn):
      self.fn = fn
    def __get__(self, obj, cls):
      value = obj.__dict__[self.fn.__name__] = self.fn(obj)
      return value
logger = ir_datasets.log.easy()


class CorpusGraph(pta.Artifact):
  def neighbours(self, docid: Union[int, str], weights: bool = False) -> Union[np.array, List[str], Tuple[np.array, np.array], Tuple[List[str], np.array]]:
    raise NotImplementedError()

  @staticmethod
  def load(path, **kwargs):
    with (Path(path)/'pt_meta.json').open('rt') as fin:
      meta = json.load(fin)
    assert meta.get('type') == 'corpus_graph'
    fmt = meta.get('format')
    if fmt == 'np_topk' or fmt == 'numpy_kmax':
      return NpTopKCorpusGraph(path, **kwargs)
    raise ValueError(f'Unknown corpus graph format: {fmt}')

  # TODO: rework & verify
  # @staticmethod
  # def from_np_dense_index(np_index, out_dir: Union[str, Path], k: int = 16, batch_size: int = 50000) -> 'CorpusGraph':
  #   out_dir = Path(out_dir)
  #   if out_dir.exists():
  #     raise FileExistsError(out_dir)
  #   out_dir.mkdir(parents=True)
  #   edges_path = out_dir/'edges.u32.np'
  #   weights_path = out_dir/'weights.f16.np'
  #   shutil.copyfile(np_index.index_path/'docnos.np', out_dir/'docnos.np')
  #   S = batch_size
  #   docnos, vectors = np_index.docnos_and_data()
  #   doc_count = 0
  #   with ir_datasets.util.finialized_file(str(edges_path), 'wb') as fe, ir_datasets.util.finialized_file(str(weights_path), 'wb') as fw:
  #     for i in logger.pbar(range(0, vectors.shape[0], S)):
  #       top_idxs = []
  #       top_scores = []
  #       left = torch.from_numpy(vectors[i:i+S]).cuda()
  #       left /= left.norm(dim=1, keepdim=True)
  #       for j in logger.pbar(range(0, vectors.shape[0], S)):
  #         right = torch.from_numpy(vectors[j:j+S]).cuda()
  #         right /= right.norm(dim=1, keepdim=True)
  #         top = (left @ right.T).topk(k+1, sorted=True, dim=1)
  #         top_idxs.append((top.indices + j).cpu())
  #         top_scores.append(top.values.cpu())
  #       top_idxs = torch.cat(top_idxs, dim=1)
  #       top_scores = torch.cat(top_scores, dim=1).cuda()
  #       top = top_scores.topk(k+1, sorted=True, dim=1)
  #       idxs = torch.gather(top_idxs, 1, top.indices.cpu())[:, 1:]
  #       sims = top.values.cpu()[:, 1:]
  #       fe.write(idxs.cpu().numpy().astype(np.uint32).tobytes())
  #       fw.write(sims.numpy().astype(np.float16).tobytes())
  #       doc_count += left.shape[0]
  #   with (out_dir/'pt_meta.json').open('wt') as fout:
  #     json.dump({
  #       'type': 'corpus_graph',
  #       'format': 'np_topk',
  #       'doc_count': doc_count,
  #       'k': k,
  #     }, fout)
  #   return NumpyKMaxCorpusGraph(out_dir)

  @staticmethod
  def from_retriever(retriever, docs_it, out_dir: Union[str, Path], k: int = 16, batch_size: int = 1024) -> 'CorpusGraph':
    out_dir = Path(out_dir)
    if out_dir.exists():
      raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    edges_path = out_dir/'edges.u32.np'
    weights_path = out_dir/'weights.f16.np'

    # First step: We need a docno <-> index mapping for this to work. Do a pass over the iterator
    # to build a docno loookup file, while also writing the contents to a temporary file that we'll read
    # in the next loop. (This avoids loading the entire iterator into memory.)
    with tempfile.TemporaryDirectory() as dout:
      with LZ4FrameFile(f'{dout}/docs.pkl.lz4', 'wb') as fout, Lookup.builder(out_dir/'docnos.npids') as docno_builder:
        for doc in logger.pbar(docs_it, miniters=1, smoothing=0, desc='first pass'):
          pickle.dump(doc, fout)
          docno_builder.add(doc['docno'])
      docnos = Lookup(out_dir/'docnos.npids')

      # Now read out everything in the file and retrieve using each one. Do this in batches for efficiency.
      with ir_datasets.util.finialized_file(str(edges_path), 'wb') as fe, ir_datasets.util.finialized_file(str(weights_path), 'wb') as fw, LZ4FrameFile(f'{dout}/docs.pkl.lz4', 'rb') as fin:
        for chunk in more_itertools.chunked(logger.pbar(range(len(docnos)), miniters=1, smoothing=0, desc='searching', total=len(docnos)), batch_size):
          chunk = [pickle.load(fin) for _ in chunk]
          res = retriever(pd.DataFrame(chunk).rename(columns={'docno': 'qid', 'text': 'query'}))
          res_by_qid = dict(iter(res.groupby('qid')))
          for docno in [c['docno'] for c in chunk]:
            did_res = res_by_qid.get(docno)
            dids, scores = [], []
            if did_res is not None:
              did_res = did_res[did_res.docno != docno].iloc[:k]
              if len(did_res) > 0:
                dids = docnos.inv[list(did_res.docno)]
                scores = list(did_res.score)
            if len(dids) < k: # if we didn't get as many as we expect, loop the document back to itself.
              dids += [docnos.inv[docno]] * (k - len(dids))
              scores += [0.] * (k - len(scores))
            fe.write(np.array(dids, dtype=np.uint32).tobytes())
            fw.write(np.array(scores, dtype=np.float16).tobytes())

    # Finally, keep track of metadata about this artefact.
    with (out_dir/'pt_meta.json').open('wt') as fout:
      json.dump({
        'type': 'corpus_graph',
        'format': 'np_topk',
        'package_hint': 'pyterrier-adaptive',
        'doc_count': len(docnos),
        'k': k,
      }, fout)

    return NpTopKCorpusGraph(out_dir)


class NpTopKCorpusGraph(CorpusGraph):
  def __init__(self, path, k=None):
    super().__init__(path)
    with (self.path/'pt_meta.json').open('rt') as fin:
      self.meta = json.load(fin)
    assert self.meta.get('type') == 'corpus_graph' and self.meta.get('format') in ('numpy_kmax', 'np_topk')
    self._data_k = self.meta['k']
    if k is not None:
      assert k <= self.meta['k']
    self._k = self.meta['k'] if k is None else k
    self._edges_path = self.path/'edges.u32.np'
    self._weights_path = self.path/'weights.f16.np'
    self._docnos = Lookup(self.path/'docnos.npids')

  def __repr__(self):
    return f'NpTopKCorpusGraph({repr(str(self.path))}, k={self._k})'

  @cached_property
  def edges_data(self):
    res = np.memmap(self._edges_path, mode='r', dtype=np.uint32).reshape(-1, self._data_k)
    if self._k != self._data_k:
      res = res[:, :self._k]
    return res

  @cached_property
  def weights_data(self):
    res = np.memmap(self._weights_path, mode='r', dtype=np.float16).reshape(-1, self._data_k)
    if self._k != self._data_k:
      res = res[:, :self._k]
    return res

  def neighbours(self, docid, weights=False):
    as_str = isinstance(docid, str)
    if as_str:
      docid = self._docnos.inv[docid]
    neigh = self.edges_data[docid]
    if as_str:
      neigh = self._docnos.fwd[neigh]
    if weights:
      weigh = self.weights_data[docid]
      return neigh, weigh
    return neigh

  def to_limit_k(self, k: int) -> 'NpTopKCorpusGraph':
    """
    Creates a version of the graph with a maximum of k edges from each node.
    k must be less than the number of edges per node from the original graph.
    """
    return NpTopKCorpusGraph(self.path, k)
