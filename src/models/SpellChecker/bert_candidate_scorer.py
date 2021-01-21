from typing import List, Tuple, Callable
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
            agg_subtoken_func: Callable = np.mean
    ):
        """Init object.

        :param bert_scorer_model: model for scoring based on Bert
        :param agg_subtoken_func: function to aggregate scores for
            WordPiece subtokens in candidates
        """
        self.bert_scorer_model = bert_scorer_model
        self.agg_subtoken_func = agg_subtoken_func

    def __call__(
            self, tokenized_sentences: List[List[str]],
            positions: List[int], candidates: List[List[str]]
    ) -> Tuple[List[int], Tuple[List[float], List[float]]]:
        """Make scoring for candidates for every sentence and adjust them.

        :param tokenized_sentences: list of tokenized sentences
        :param positions: positions for candidates scoring for each sentence
        :param candidates: candidates for given positions
            in each sentence

        :returns:
            indices of best candidates for each position
            (list of current scores for each sentence,
            list of best scores for each sentence)
        """
        # add mask tokens to given positions
        masked_tokenized_sentences = []
        for i, pos in enumerate(positions):
            current_sentence = copy(tokenized_sentences[i])
            current_sentence[pos] = self.bert_scorer_model.tokenizer.mask_token
            masked_tokenized_sentences.append(current_sentence)

        # detokenize sentences
        # it is made by join because there is problem with MosesDetokenizer
        # WordPiece tokenizer can't see [MASK] token in "[MASK]?" string
        masked_sentences = [
            ' '.join(sentence) for sentence in masked_tokenized_sentences
        ]

        # make scoring
        scoring_results = self.bert_scorer_model(
            masked_sentences, candidates, agg_func=self.agg_subtoken_func
        )

        # find best corrections
        current_scores = [
            scoring_results[idx][0]
            for idx in range(len(scoring_results))
        ]
        best_scores_with_indices = [
            max(enumerate(sentence_scoring_results), key=lambda x: x[1])
            for sentence_scoring_results in scoring_results
        ]
        best_indices = [x[0] for x in best_scores_with_indices]
        best_scores = [x[1] for x in best_scores_with_indices]

        return best_indices, (current_scores, best_scores)
