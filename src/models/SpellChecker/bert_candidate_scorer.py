from typing import List, Callable
from copy import copy

import numpy as np

from src.models.BertScorer.bert_scorer_correction import BertScorerCorrection


class BertCandidateScorer:
    """Wrapper class over BertScorerCorrection
    to score candidates for correction.
    """

    def __init__(
            self,
            bert_scorer_model: BertScorerCorrection,
            agg_subtoken_func: Callable = np.sum
    ):
        self.bert_scorer_model = bert_scorer_model
        self.agg_subtoken_func = agg_subtoken_func

    def __call__(
            self, tokenized_sentences: List[List[str]],
            positions: List[int], candidates: List[List[str]],
            detokenizer: Callable
    ) -> List[List[float]]:
        """Make scoring for candidates for every sentence and adjust them.

        :param tokenized_sentences: list of tokenized sentences
        :param positions: positions for candidates scoring for each sentence
        :param candidates: candidates for given positions
            in each sentence
        :param detokenizer: object to make sentence
            string from tokenized sentence

        :returns: score for each candidate for each sentence
        """
        # TODO: add processing case for candidates
        # add mask tokens to given positions
        masked_tokenized_sentences = []
        for i, pos in enumerate(positions):
            current_sentence = copy(tokenized_sentences[i])
            current_sentence[pos] = self.bert_scorer_model.tokenizer.mask_token
            masked_tokenized_sentences.append(current_sentence)

        # detokenize sentences
        # it is made by join because there is problem with MosesDetokenizer
        # WordPiece tokenizer can't see [MASK]?
        masked_sentences = [
            ' '.join(sentence) for sentence in masked_tokenized_sentences
        ]

        # make scoring
        scoring_results = self.bert_scorer_model(
            masked_sentences, candidates, agg_func=self.agg_subtoken_func
        )

        # adjust scoring results
        # now it is log probabilities, that makes them negative
        # make them positive
        adjusted_results = [
            [-1/result for result in results]
            for results in scoring_results
        ]

        return adjusted_results
