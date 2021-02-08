from typing import List, Tuple, Dict, Any, Optional, Callable
from copy import copy

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from src.models.BertScorer import BertScorerCorrection

import catboost


class BertScorer:
    """Wrapper class over BertScorerCorrection
    to score candidates for correction.
    """

    def __init__(
            self,
            bert_scorer_model: BertScorerCorrection,
            agg_subtoken_func: str = 'mean'
    ):
        """Init object.

        :param bert_scorer_model: model for scoring based on Bert
        :param agg_subtoken_func: function name to aggregate scores for
            WordPiece subtokens in candidates
        """
        self.bert_scorer_model = bert_scorer_model
        if agg_subtoken_func not in ['len', 'sum', 'mean']:
            raise ValueError(
                'Parameter agg_subtoken_func should have one of '
                'these values: len, sum, mean'
            )
        self.agg_subtoken_func = agg_subtoken_func

    def __call__(
            self, tokenized_sentences: List[List[str]],
            positions: List[int],
            candidates: List[List[Dict[str, Any]]],
            return_scoring_info: bool = False
    ) -> Tuple[List[List[Any]], Optional[List[List[Dict[str, Any]]]]]:
        """Make scoring for candidates for every sentence and adjust them.

        :param tokenized_sentences: list of tokenized sentences
        :param positions: positions for candidates scoring for each sentence
        :param candidates: candidates and their features
            for given positions in each sentence
        :param return_scoring_info: return or not return additional
            information about scoring

        :returns:
            results of scoring,
            additional information about scoring
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
        candidates_tokens = [[x['token'] for x in candidates_sentence]
                             for candidates_sentence in candidates]
        # make scoring
        scoring_info_raw = self.bert_scorer_model(
            masked_sentences, candidates_tokens
        )

        # process scoring info
        scoring_info: Optional[List[List[Dict[str, Any]]]] = []
        scoring_results = []
        for sentence_info_raw in scoring_info_raw:
            sentence_info = []
            scoring_results_sentence = []
            for candidate_info_raw in sentence_info_raw:
                candidate_info = {
                    'bert_score_len': len(candidate_info_raw),
                    'bert_score_sum': sum(candidate_info_raw),
                    'bert_score_mean': (
                            sum(candidate_info_raw)/len(candidate_info_raw)
                    )
                }
                sentence_info.append(candidate_info)
                scoring_results_sentence.append(
                    candidate_info[f'bert_score_{self.agg_subtoken_func}']
                )
            scoring_info.append(sentence_info)
            scoring_results.append(scoring_results_sentence)

        if not return_scoring_info:
            scoring_info = None
        return scoring_results, scoring_info


# TODO: refactor this
#   it may be better to create class MLScorer
#   and pass some predict function inside
#   now there is a lot of duplicate code
class SVMScorer:
    """SVM model trained on candidate features features."""

    def __init__(
            self,
            svm_model: Pipeline,
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
            candidates: List[List[Dict[str, Any]]],
            return_scoring_info: bool = False
    ) -> Tuple[List[List[float]], Optional[Any]]:
        """Make scoring for candidates for every sentence and adjust them.

        :param tokenized_sentences: list of tokenized sentences
        :param positions: positions for candidates scoring for each sentence
        :param candidates: candidates and their features
            for given positions in each sentence
        :param return_scoring_info: return or not return additional
            information about scoring

        :returns:
            results of scoring,
            additional information about scoring
        """
        candidates_new = copy(candidates)
        # make scoring using Bert if possible
        if self.bert_scorer:
            # make scoring with getting additional info for features
            _, bert_scoring_info = self.bert_scorer(
                tokenized_sentences, positions, candidates,
                return_scoring_info=True
            )
            # add this features to candidates_features
            for num_sent, candidates_sentence in enumerate(candidates_new):
                for i, candidate in enumerate(candidates_sentence):
                    candidate.update(bert_scoring_info[num_sent][i])

        # process each sentence separately
        scoring_results = []
        scoring_info = []
        for num_sent, candidates_sentence in enumerate(
                candidates_new
        ):
            # prepare data for scoring with SVM model
            data_list = copy(candidates_sentence)
            data_df = pd.DataFrame(data_list)
            data_df.drop(columns=['token'], inplace=True)
            data_df = data_df.astype(float)
            if return_scoring_info:
                scoring_info.append(data_list)
            scoring_results.append(
                self.svm_model.decision_function(data_df).tolist()
            )

        if not return_scoring_info:
            scoring_info = None
        return scoring_results, scoring_info


class CatBoostScorer:
    """CatBoost model trained on candidate features features."""

    def __init__(
            self,
            catboost_model: catboost.CatBoost,
            bert_scorer: Optional[BertScorer] = None
    ):
        """Init object.

        :param catboost_model: catboost model for scoring
        :param bert_scorer: scorer based on Bert
        """
        self.catboost_model = catboost_model
        self.bert_scorer = bert_scorer

    def __call__(
            self, tokenized_sentences: List[List[str]],
            positions: List[int],
            candidates: List[List[Dict[str, Any]]],
            return_scoring_info: bool = False
    ) -> Tuple[List[List[float]], Optional[Any]]:
        """Make scoring for candidates for every sentence and adjust them.

        :param tokenized_sentences: list of tokenized sentences
        :param positions: positions for candidates scoring for each sentence
        :param candidates: candidates and their features
            for given positions in each sentence
        :param return_scoring_info: return or not return additional
            information about scoring

        :returns:
            results of scoring,
            additional information about scoring
        """
        # TODO: make abstract class with processing other scorers
        candidates_new = copy(candidates)
        # make scoring using Bert if possible
        if self.bert_scorer:
            # make scoring with getting additional info for features
            _, bert_scoring_info = self.bert_scorer(
                tokenized_sentences, positions, candidates,
                return_scoring_info=True
            )
            # add this features to candidates_features
            for num_sent, candidates_sentence in enumerate(candidates_new):
                for i, candidate in enumerate(candidates_sentence):
                    candidate.update(bert_scoring_info[num_sent][i])

        # process each sentence separately
        scoring_results = []
        scoring_info = []
        for num_sent, candidates_sentence in enumerate(
                candidates_new
        ):
            # prepare data for scoring with CatBoost model
            data_list = copy(candidates_sentence)
            data_df = pd.DataFrame(data_list)
            data_df.drop(columns=['token'], inplace=True)
            data_df = data_df.astype(float)
            if return_scoring_info:
                scoring_info.append(data_list)
            data_pool = catboost.Pool(data=data_df)
            scoring_results.append(
                self.catboost_model.predict(data_pool).tolist()
            )

        if not return_scoring_info:
            scoring_info = None
        return scoring_results, scoring_info
