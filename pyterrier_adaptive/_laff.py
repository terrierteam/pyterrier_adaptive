from typing import Optional, Union, List
import shutil
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import count
from more_itertools import chunked
import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_adaptive import NpTopKCorpusGraph


class Laff(pt.Transformer):
    """A transformer that computes a learned affinity score between two document texts using a transformer model. """

    def __init__(self,
        model: str = 'macavaney/laff',
        *,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 128,
        max_length: int = 512,
        verbose: bool = False
    ):
        """ Initialize the LAFF transformer.

        Args:
            model: the name of the transformer model to use.
            device: the device to use for the transformer model.
            batch_size: the batch size to use for processing.
            max_length: the maximum length of the input text.
            verbose: whether to display progress bars.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.verbose = verbose

    def compute_affinity(self,
        texts_left: Union[str, List[str]],
        texts_right: Union[str, List[str]]
    ) -> List[float]:
        """ Compute the affinity scores between two lists of texts.

        Args:
            texts_left: the left-hand side texts.
            texts_right: the right-hand side texts.

        Returns:
            A list of affinity scores.
        """
        if isinstance(texts_left, str) and isinstance(texts_right, str):
            return self.compute_affinity([texts_left], [texts_right])[0]
        elif isinstance(texts_left, str):
            texts_left = [texts_left] * len(texts_right)
        elif isinstance(texts_right, str):
            texts_right = [texts_right] * len(texts_left)
        assert len(texts_left) == len(texts_right)

        affinity_scores = []

        it = chunked(zip(texts_left, texts_right), self.batch_size)
        if self.verbose:
            it = tqdm(it, unit='batch', total=len(texts_left))

        for batch in it:
            batch = list(batch)
            batch_left, batch_right = zip(*batch)
            enc = self.tokenizer(
                list(batch_left),
                list(batch_right),
                max_length=self.max_length,
                padding='longest',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            ).to(self.device)

            with torch.no_grad(), torch.autocast(device_type=self.device.type):
                outputs = self.model(**enc)
                affinity_scores += outputs.logits.flatten().tolist()

        return affinity_scores

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """ Compute the affinity scores between two columns of texts in the input.

        Expects columns ``text`` (for the left-side text) and ``other_text`` (for the right-side text).

        Results are sorted by the left-side text and the affinity score.
        """
        pta.validate.columns(inp, includes=['text', 'other_text'])
        affinity = self.compute_affinity(inp['text'], inp['other_text'])
        res = inp.assign(affinity=affinity)
        res.sort_values(['text', 'affinity'], ascending=[True, False], inplace=True)
        return res

    def apply_to_graph(self,
        graph: NpTopKCorpusGraph,
        text_loader: pt.Transformer,
        out_path: Optional[str] = None,
        *,
        verbose: Optional[bool] = None
    ) -> NpTopKCorpusGraph:
        """ Apply the LAFF transformer to a corpus graph to construct a new one.

        Args:
            graph: the input corpus graph.
            text_loader: a transformer that loads the text for a given document.
            out_path: the path to save the output corpus graph. If not provided, the input graph's path is used with a '.laff' extension.
            verbose: whether to display progress bars.
        """
        if out_path is None:
            out_path = str(graph.path) + '.laff'
        if verbose is None:
            verbose = self.verbose

        with pta.ArtifactBuilder(NpTopKCorpusGraph, path=out_path) as b:
            shutil.copyfile(graph.path / 'docnos.npids', b.path / 'docnos.npids')
            docnos = graph._docnos
            doc_count = b.metadata['doc_count'] = len(docnos)
            k = b.metadata['k'] = graph._k
            with (b.path/'edges.u32.np').open('wb') as fe, (b.path/'weights.f16.np').open('wb') as fw:
                it = zip(count(), docnos, graph.edges_data)
                if verbose:
                    it = tqdm(it, unit='doc', total=doc_count)
                for did, docno, neighbor_ids in it:
                    neighbor_ids = neighbor_ids[neighbor_ids != did] # ignore self-links
                    texts = text_loader(pd.DataFrame({'docno': [docno] + list(docnos.fwd[neighbor_ids])}))['text']
                    this_text, other_texts = texts[0], texts[1:]
                    affinity = np.array(self.compute_affinity(this_text, other_texts))
                    sort = np.argsort(-affinity)
                    weights = affinity[sort].astype(np.float16)
                    edges = neighbor_ids[sort].astype(np.uint32)
                    if len(weights) < k: # pad up to k with self-links if needed
                        weights = np.pad(weights, (0, k - len(weights)), constant_values=float('-inf'))
                        edges = np.pad(edges, (0, k - len(edges)), constant_values=did)
                    fe.write(edges.tobytes())
                    fw.write(weights.tobytes())

        return NpTopKCorpusGraph(out_path)
