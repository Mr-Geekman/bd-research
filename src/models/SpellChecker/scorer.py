from typing import List, Tuple, Dict, Any, Optional, Callable
from copy import copy

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from src.models.BertScorer import BertScorerCorrection


class BertScorer:
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
            positions: List[int],
            candidates_features: List[List[Tuple[str, Dict[str, Any]]]]
    ) -> List[List[float]]:
        """Make scoring for candidates for every sentence and adjust them.

        :param tokenized_sentences: list of tokenized sentences
        :param positions: positions for candidates scoring for each sentence
        :param candidates_features: candidates and their features
            for given positions in each sentence

        :returns: results of scoring
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

        # get only tokens of candidates
        candidates = [[x[0] for x in candidates_sentence]
                      for candidates_sentence in candidates_features]
        # make scoring
        scoring_results = self.bert_scorer_model(
            masked_sentences, candidates, agg_func=self.agg_subtoken_func
        )
        return scoring_results


class SVMScorer:
    """SVM model trained on candidate features features."""

    def __init__(
            self,
            svm_model: LinearSVC,
            bert_scorer: Optional[BertScorer] = None
    ):
        """Init object.

        :param svm_model: svm model for scoring
        :param bert_scorer: scorer based on Bert
        """
        self.svm_model = svm_model
        self.bert_scorer = bert_scorer
        # TODO: can be added list of additional scorers with their names

    def __call__(
            self, tokenized_sentences: List[List[str]],
            positions: List[int],
            candidates_features: List[List[Tuple[str, Dict[str, Any]]]]
    ) -> List[List[float]]:
        """Make scoring for candidates for every sentence and adjust them.

        :param tokenized_sentences: list of tokenized sentences
        :param positions: positions for candidates scoring for each sentence
        :param candidates_features: candidates and their features
            for given positions in each sentence

        :returns: results of scoring
        """
        candidates_features_new = copy(candidates_features)
        # make scoring using Bert if possible
        if self.bert_scorer:
            # make scoring
            bert_scoring_results = self.bert_scorer(
                tokenized_sentences, positions, candidates_features
            )
            # add this features to candidates_features
            for num_sent, candidates_sentence in enumerate(
                    candidates_features_new
            ):
                for i, candidate in enumerate(candidates_sentence):
                    candidate[1]['bert_score'] = bert_scoring_results[
                        num_sent
                    ][i]

        # process each sentence separately
        scoring_results = []
        for num_sent, candidates_features_sentence in enumerate(
                candidates_features_new
        ):
            # prepare data for scoring with SVM model
            data_dict = [x[1] for x in candidates_features_sentence]
            data = pd.DataFrame(data_dict).astype(float)
            scoring_results.append(
                self.svm_model.decision_function(data).tolist()
            )
        return scoring_results
