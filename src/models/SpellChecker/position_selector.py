import re
from typing import List, Tuple, Set, Dict, Optional, Callable, Any
from string import punctuation

import numpy as np
import kenlm


class KenlmPositionSelector:
    """Class that selects best positions for correction."""

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
        self.left_right_model = left_right_model
        self.right_left_model = right_left_model
        self.margin_border = margin_border
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

            # calculate aggregated score for each candidate for each position
            for pos, candidates_position in enumerate(candidates[num_sent]):
                for candidate in candidates_position:
                    left_right_score = candidate['kenlm_left_right_score']
                    right_left_score = candidate['kenlm_right_left_score']
                    candidate['kenlm_agg_score'] = 2 * (
                        left_right_score * right_left_score
                    ) / (left_right_score + right_left_score)

            # calculate margin of aggregated score between candidate
            # and current candidate in this position
            for pos, candidates_position in enumerate(candidates[num_sent]):
                for candidate in candidates_position:
                    candidate['margin_kenlm_agg'] = (
                        candidate['kenlm_agg_score'] -
                        candidates_position[0]['kenlm_agg_score']
                    )

            best_position_score = 0
            best_position = None
            for pos, candidates_position in enumerate(candidates[num_sent]):
                # skip black list with positions (if possible)
                if pos not in positions_black_lists[num_sent]:
                    position_score = max([
                        candidate['margin_kenlm_agg']
                        for candidate in candidates_position
                    ])
                    if position_score >= best_position_score:
                        best_position = pos
                        best_position_score = position_score

            best_positions.append(best_position)
            if best_position is None:
                # we should stop correction of this sentence because
                # there is no positions that not in black list
                criteria_results.append(True)
                continue
            else:
                criteria_results.append(
                    best_position_score <= self.margin_border
                )

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
