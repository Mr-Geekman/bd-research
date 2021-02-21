import re
from abc import ABC
from typing import List, Tuple, Set, Dict, Optional, Callable, Any
from string import punctuation
from collections import defaultdict

import numpy as np
import pandas as pd
import kenlm


class KenlmBasePositionSelector(ABC):
    """Abstract base class that selects best positions
    for correction based on kenlm scores.
    """

    def __init__(
            self, left_right_model: kenlm.Model, right_left_model: kenlm.Model,
            agg_subtoken_func: Callable = sum,
    ):
        """Init object.

        :param left_right_model: kenlm language model from left to right
        :param right_left_model: kenlm language model from right to left
        :param agg_subtoken_func: function to aggregate scores
            for few words in a candidate

        """
        self.left_right_model = left_right_model
        self.right_left_model = right_left_model
        self.agg_subtoken_func = agg_subtoken_func

    def __call__(
            self, tokenized_sentences_raw: List[List[str]],
            candidates_raw: List[List[List[Dict[str, Any]]]],
            positions_black_lists_raw: List[Set[int]] = None
    ) -> Tuple[List[Optional[int]],
               List[bool]]:
        """Select best positions for correction and candidates to score.

        :param tokenized_sentences_raw: list of sentences,
            that are list of tokens
        :param candidates_raw: list of possible corrections and their features
            for each position in each sentence
        :param positions_black_lists_raw: black list of positions
            for each sentence

        :returns:
            list of best positions for each sentence
                (or None if all positions are unavailable)
            results of stopping criteria
        """
        if positions_black_lists_raw is None:
            positions_black_lists_raw = [
                set() for _ in range(len(tokenized_sentences_raw))
            ]

        # preprocess sentences
        tokenized_sentences = []
        indices_mappings = []
        for sentence in tokenized_sentences_raw:
            processed_sentence, indices_mapping = self._preprocess_sentence(
                sentence
            )
            tokenized_sentences.append(processed_sentence)
            indices_mappings.append(indices_mapping)

        # prune some lists according to indices_mappings
        candidates = [
            [candidates_raw[i][idx] for idx in indices_mapping]
            for i, indices_mapping in enumerate(indices_mappings)
        ]
        positions_black_lists = [
            {j for j, idx in enumerate(indices_mapping)
             if idx in positions_black_lists_raw[i]}
            for i, indices_mapping in enumerate(indices_mappings)
        ]

        # process each sentence separately
        best_positions = []
        criteria_results = []
        for num_sent, tokenized_sentence in enumerate(tokenized_sentences):
            # add kenlm scores to candidates
            self._add_kenlm_scores(
                tokenized_sentence, candidates[num_sent]
            )

            best_position, criteria_result = self._predict_sentence(
                candidates[num_sent], positions_black_lists[num_sent]
            )
            best_positions.append(best_position)
            criteria_results.append(criteria_result)

        # correct best_positions according to initial_mappings
        best_positions_adjusted = []
        for best_position, indices_mapping in zip(best_positions,
                                                  indices_mappings):
            if best_position is None:
                best_positions_adjusted.append(None)
            else:
                best_positions_adjusted.append(indices_mapping[best_position])

        return best_positions_adjusted, criteria_results

    def _preprocess_sentence(
            self, tokenized_sentence: List[str]
    ) -> Tuple[List[str], List[int]]:
        """Preprocess tokenized sentence and create mapping of indices.

        :param tokenized_sentence: list of tokens

        :returns: processed tokenized sentence,
            mapping of current indices to initial
        """
        # remove punctuation from tokens and make mapping to initial indices
        processed_sentence = []
        indices_mapping = []
        for i, token in enumerate(tokenized_sentence):
            if not re.fullmatch(f'[{punctuation}]+', token):
                processed_sentence.append(token)
                indices_mapping.append(i)

        return processed_sentence, indices_mapping

    def _add_kenlm_scores(
            self,
            tokenized_sentence: List[str],
            candidates: List[List[Dict[str, Any]]]
    ) -> None:
        """Add scores of kenlm model to candidates

        :param tokenized_sentence: list of tokens
        :param candidates: list of possible corrections and their features
            for sentence
        """
        # make left-to-right pass
        state = kenlm.State()
        self.left_right_model.BeginSentenceWrite(state)
        prev_state = state
        for pos, token in enumerate(tokenized_sentence):
            for candidate in candidates[pos]:
                candidate_token = candidate['token']
                candidate_lm_scores = []
                prev_local_state = prev_state
                # process subtokens if candidate contains whitespaces
                for candidate_subtoken in candidate_token.split():
                    state = kenlm.State()
                    candidate_lm_scores.append(
                        self.left_right_model.BaseScore(
                            prev_local_state, candidate_subtoken, state
                        )
                    )
                    prev_local_state = state
                # add feature to candidates
                candidate['kenlm_left_right_score'] = self.agg_subtoken_func(
                    candidate_lm_scores
                )

            # find prev state according to current token
            state = kenlm.State()
            self.left_right_model.BaseScore(
                prev_state,
                candidates[pos][0]['token'], state
            )
            prev_state = state

        # make right-to-left pass
        state = kenlm.State()
        self.right_left_model.BeginSentenceWrite(state)
        prev_state = state
        for inv_pos, token in enumerate(tokenized_sentence[::-1]):
            pos = len(tokenized_sentence) - 1 - inv_pos
            for i, candidate in enumerate(
                    candidates[pos]
            ):
                candidate_token = candidate['token']
                candidate_lm_scores = []
                prev_local_state = prev_state
                # process subtokens if candidate contains whitespaces
                for candidate_subtoken in candidate_token.split()[::-1]:
                    state = kenlm.State()
                    candidate_lm_scores.append(
                        self.right_left_model.BaseScore(
                            prev_local_state, candidate_subtoken, state
                        )
                    )
                    prev_local_state = state
                # add feature to candidates
                candidate['kenlm_right_left_score'] = self.agg_subtoken_func(
                    candidate_lm_scores
                )

            # find prev state according to current token
            state = kenlm.State()
            self.right_left_model.BaseScore(
                prev_state,
                candidates[pos][0]['token'], state
            )
            prev_state = state

        # calculate aggregated score for each candidate for each position
        for pos, candidates_position in enumerate(candidates):
            for candidate in candidates_position:
                left_right_score = candidate['kenlm_left_right_score']
                right_left_score = candidate['kenlm_right_left_score']
                candidate['kenlm_agg_score'] = 2 * (
                    left_right_score * right_left_score
                ) / (left_right_score + right_left_score)

        # calculate margin of aggregated score between candidate
        # and current candidate in this position
        for pos, candidates_position in enumerate(candidates):
            for candidate in candidates_position:
                candidate['margin_kenlm_agg'] = (
                    candidate['kenlm_agg_score'] -
                    candidates_position[0]['kenlm_agg_score']
                )

    def _predict_sentence(
            self,
            candidates: List[List[Dict[str, Any]]],
            positions_black_list: Set[int]
    ) -> Tuple[Optional[int], bool]:
        """Predict results of model for sentence: position for correction and
        indicator of stopping criteria.

        :param candidates: list of possible corrections and their features
            for sentence
        :param positions_black_list: black list of positions for sentence

        :return:
            best position for correction
            indicator of stopping
        """
        pass


class KenlmMarginPositionSelector(KenlmBasePositionSelector):
    """Class that selects best positions for correction
    based on margin aggregated kenlm score
    between current token and best token.
    """

    def __init__(
            self, left_right_model: kenlm.Model, right_left_model: kenlm.Model,
            margin_border: float,
            agg_subtoken_func: Callable = sum
    ):
        """Init object.

        :param left_right_model: kenlm language model from left to right
        :param right_left_model: kenlm language model from right to left
        :param margin_border: border for stopping criteria
        :param agg_subtoken_func: function to aggregate scores
            for few words in a candidate

        """
        super().__init__(
            left_right_model, right_left_model,
            agg_subtoken_func=agg_subtoken_func
        )
        self.margin_border = margin_border

    def _predict_sentence(
            self,
            candidates: List[List[Dict[str, Any]]],
            positions_black_list: Set[int]
    ) -> Tuple[Optional[int], bool]:
        """Predict results of model for sentence: position for correction and
        indicator of stopping criteria.

        :param candidates: list of possible corrections and their features
            for sentence
        :param positions_black_list: black list of positions for sentence

        :return:
            best position for correction
            indicator of stopping
        """
        best_position_score = 0
        best_position = None
        position_scores = [] # for debug
        for pos, candidates_position in enumerate(candidates):
            # skip black list with positions (if possible)
            if pos not in positions_black_list:
                position_score = max([
                    candidate['margin_kenlm_agg']
                    for candidate in candidates_position
                ])
                position_scores.append(position_score)
                if position_score >= best_position_score:
                    best_position = pos
                    best_position_score = position_score

        if best_position is None:
            return best_position, True
        else:
            return best_position, best_position_score <= self.margin_border


class KenlmMLPositionSelector(KenlmBasePositionSelector):
    """Class that selects best positions for correction based on prediction
    of ML model.
    """

    def __init__(
            self, left_right_model: kenlm.Model, right_left_model: kenlm.Model,
            ml_model,
            agg_subtoken_func: Callable = sum
    ):
        """Init object.

        :param left_right_model: kenlm language model from left to right
        :param right_left_model: kenlm language model from right to left
        :param ml_model: model for classification, should predict require
            position correction or not and have predict proba
            for sorting confidence of prediction
        :param agg_subtoken_func: function to aggregate scores
            for few words in a candidate

        """
        super().__init__(
            left_right_model, right_left_model,
            agg_subtoken_func=agg_subtoken_func
        )
        self.ml_model = ml_model

    def _predict_sentence(
            self,
            candidates: List[List[Dict[str, Any]]],
            positions_black_list: Set[int]
    ) -> Tuple[Optional[int], bool]:
        """Predict results of model for sentence: position for correction and
        indicator of stopping criteria.

        :param candidates: list of possible corrections and their features
            for sentence
        :param positions_black_list: black list of positions for sentence

        :return:
            best position for correction
            indicator of stopping
        """
        # create features for ML model
        df_dict = defaultdict(lambda: list())
        for pos, candidates_position in enumerate(candidates):
            candidate = candidates_position[0]
            df_dict['is_title'].append(candidate['is_title'])
            df_dict['is_upper'].append(candidate['is_upper'])
            df_dict['is_lower'].append(candidate['is_lower'])
            df_dict['is_first'].append(candidate['is_first'])
            df_dict['is_title_not_is_first'].append(
                candidate['is_title'] and not candidate['is_first']
            )

            df_dict['num_candidates'].append(len(candidates_position))

            current_candidate = next((
                candidate for candidate in candidates_position
                if candidate['is_current']
            ))
            df_dict['is_current_contains_hyphen'].append(
                current_candidate['contains_hyphen']
            )
            df_dict['current_kenlm_left_right_score'].append(
                current_candidate['kenlm_left_right_score']
            )
            df_dict['current_kenlm_right_left_score'].append(
                current_candidate['kenlm_right_left_score']
            )
            df_dict['current_kenlm_agg_score'].append(
                current_candidate['kenlm_agg_score']
            )

            from_levenshtein = []
            from_phonetic = []
            from_handcode = []
            kenlm_left_right_better = []
            kenlm_right_left_better = []
            kenlm_agg_better = []
            margin_kenlm_left_right = 0
            margin_kenlm_right_left = 0
            margin_kenlm_agg = 0
            for candidate in candidates_position:
                from_levenshtein.append(candidate['from_levenshtein_searcher'])
                from_phonetic.append(candidate['from_phonetic_searcher'])
                from_handcode.append(candidate['from_handcode_searcher'])
                kenlm_left_right_better.append(
                    candidate['kenlm_left_right_score']
                    > current_candidate['kenlm_left_right_score']
                )
                kenlm_right_left_better.append(
                    candidate['kenlm_right_left_score']
                    > current_candidate['kenlm_right_left_score']
                )
                kenlm_agg_better.append(
                    candidate['kenlm_agg_score']
                    > current_candidate['kenlm_agg_score']
                )
                margin_kenlm_left_right = max(
                    margin_kenlm_left_right,
                    candidate['kenlm_left_right_score']
                    - current_candidate['kenlm_left_right_score']
                )
                margin_kenlm_right_left = max(
                    margin_kenlm_right_left,
                    candidate['kenlm_right_left_score']
                    - current_candidate['kenlm_right_left_score']
                )
                margin_kenlm_agg = max(
                    margin_kenlm_agg,
                    candidate['kenlm_agg_score']
                    - current_candidate['kenlm_agg_score']
                )

            df_dict['freq_levenshtein_searcher'].append(
                np.mean(from_levenshtein)
            )
            df_dict['freq_phonetic_searcher'].append(np.mean(from_phonetic))
            df_dict['freq_handcode_searcher'].append(np.mean(from_handcode))
            df_dict['freq_kenlm_left_right_better_current'].append(
                np.mean(kenlm_left_right_better)
            )
            df_dict['freq_kenlm_right_left_better_current'].append(
                np.mean(kenlm_right_left_better)
            )
            df_dict['freq_kenlm_agg_better_current'].append(
                np.mean(kenlm_agg_better)
            )
            df_dict['margin_kenlm_left_right_score'].append(
                margin_kenlm_left_right
            )
            df_dict['margin_kenlm_right_left_score'].append(
                margin_kenlm_right_left
            )
            df_dict['margin_kenlm_agg_score'].append(
                margin_kenlm_agg
            )

        df = pd.DataFrame(df_dict)
        # predicts should we do correction or not
        predicted = self.ml_model.predict(df)
        # get logits of prediction for sorting
        logits = self.ml_model.decision_function(df)

        best_position_score = 0
        best_position = None
        for pos, candidates_position in enumerate(candidates):
            # skip black list with positions (if possible)
            if pos not in positions_black_list:
                position_score = logits[pos]
                if position_score >= best_position_score:
                    best_position = pos
                    best_position_score = position_score

        if best_position is None:
            return best_position, True
        else:
            # stop correction if best position should not be corrected
            # according to the model
            return best_position, not predicted[best_position]
