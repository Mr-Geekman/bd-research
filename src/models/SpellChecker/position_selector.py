import re
from typing import List, Tuple, Set, Optional, Callable
from string import punctuation

import numpy as np
import kenlm


class KenlmPositionSelector:
    """Class that selects best positions for correction."""

    def __init__(
            self, left_right_model: kenlm.Model, right_left_model: kenlm.Model,
            agg_subtoken_func: Callable = np.sum
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
            candidates_raw: List[List[List[str]]],
            current_tokens_candidates_indices_raw: List[List[int]],
            num_selected_candidates: Optional[int] = None,
            positions_black_lists_raw: List[Set[int]] = None
    ) -> Tuple[List[Optional[int]],
               Tuple[List[Optional[float]], List[Optional[float]]],
               List[List[str]]]:
        """Select best positions for correction and candidates to score.

        :param tokenized_sentences_raw: list of sentences,
            that are list of tokens
        :param candidates_raw: list of possible corrections
            for each position in each sentence
        :param current_tokens_candidates_indices_raw: list of indices
            for current tokens in sentences
        :param num_selected_candidates: maximum number of selected tokens
            for each position in sentence or no restriction
        :param positions_black_lists_raw: black list of positions
            for each sentence

        :returns:
            list of best positions for each sentence
                (or None if all positions are unavailable)
            (list of current scores for best positions,
            list of best predicted scores for best positions)
                (or None if all positions are unavailable)
            selected candidates for each position for each sentence
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
        current_tokens_candidates_indices = [
            [current_tokens_candidates_indices_raw[i][idx]
             for idx in indices_mapping]
            for i, indices_mapping in enumerate(indices_mappings)
        ]
        positions_black_lists = [
            {j for j, idx in enumerate(indices_mapping)
             if idx in positions_black_lists_raw[i]}
            for i, indices_mapping in enumerate(indices_mappings)
        ]

        # process each sentence separately
        best_positions = []
        positions_candidates = []
        # current scores for best positions
        positions_scores_cur = []
        # predicted best scores for best positions
        positions_scores_pred = []
        for num_sent, tokenized_sentence in enumerate(tokenized_sentences):
            sentence_candidates_scores = [
                [] for _ in range(len(tokenized_sentence))
            ]
            # make left-to-right pass
            state = kenlm.State()
            self.left_right_model.BeginSentenceWrite(state)
            prev_state = state
            for pos, token in enumerate(tokenized_sentence):
                for candidate in candidates[num_sent][pos]:
                    candidate_lm_score = []
                    prev_local_state = prev_state
                    # process subtokens if candidate contains whitespaces
                    for candidate_subtoken in candidate.split():
                        state = kenlm.State()
                        candidate_lm_score.append(
                            self.left_right_model.BaseScore(
                                prev_local_state, candidate_subtoken, state
                            )
                        )
                        prev_local_state = state
                    sentence_candidates_scores[pos].append(
                        [self.agg_subtoken_func(candidate_lm_score)]
                    )

                # find prev state according to current token
                state = kenlm.State()
                self.left_right_model.BaseScore(
                    prev_state,
                    candidates[num_sent][pos][
                        current_tokens_candidates_indices[num_sent][pos]
                    ], state
                )
                prev_state = state

            # make right-to-left pass
            state = kenlm.State()
            self.right_left_model.BeginSentenceWrite(state)
            prev_state = state
            for inv_pos, token in enumerate(tokenized_sentence[::-1]):
                pos = len(tokenized_sentence) - 1 - inv_pos
                for i, candidate in enumerate(
                        candidates[num_sent][pos]
                ):
                    candidate_lm_score = []
                    prev_local_state = prev_state
                    # process subtokens if candidate contains whitespaces
                    for candidate_subtoken in candidate.split()[::-1]:
                        state = kenlm.State()
                        candidate_lm_score.append(
                            self.right_left_model.BaseScore(
                                prev_local_state, candidate_subtoken, state
                            )
                        )
                        prev_local_state = state
                    sentence_candidates_scores[pos][i].append(
                        self.agg_subtoken_func(candidate_lm_score)
                    )

                # find prev state according to current token
                state = kenlm.State()
                self.right_left_model.BaseScore(
                    prev_state,
                    candidates[num_sent][pos][
                        current_tokens_candidates_indices[num_sent][pos]
                    ], state
                )
                prev_state = state

            # calculate final score for each candidate for each position
            # TODO: unite code blocks below in one block after testing model
            # TODO: rewrite using numpy
            #   (current problem is that lengths
            #   aren't the same in different lists)
            sentence_candidates_agg_scores = [
                [
                    self._calculate_position_score(
                        pos, len(tokenized_sentence) - 1 - pos, tuple(score)
                    )
                    for score in scores
                ]
                for pos, scores in enumerate(sentence_candidates_scores)
            ]

            # add to the score score of current token
            sentence_candidates_scores_pairs = []
            for pos, scores in enumerate(sentence_candidates_agg_scores):
                sentence_position_candidates_scores_pairs = []
                for j, score in enumerate(scores):
                    sentence_position_candidates_scores_pairs.append((
                        scores[
                            current_tokens_candidates_indices[num_sent][pos]
                        ],
                        score
                    ))
                sentence_candidates_scores_pairs.append(
                    sentence_position_candidates_scores_pairs
                )

            # select best position
            max_positions_scores = [
                max(pairs, key=lambda x: x[1] - x[0])
                for pairs in sentence_candidates_scores_pairs
            ]
            positions_with_scores = sorted(
                enumerate(max_positions_scores),
                key=lambda x: x[1][1] - x[1][0],
                reverse=True
            )

            # skip black list with positions (if possible)
            best_position_score = (None, (None, None))
            for position_with_score in positions_with_scores:
                if position_with_score[0] not in positions_black_lists[num_sent]:
                    best_position_score = position_with_score
                    break
            best_position = best_position_score[0]
            best_positions.append(best_position_score[0])
            positions_scores_cur.append(best_position_score[1][0])
            positions_scores_pred.append(best_position_score[1][1])
            if best_position is None:
                positions_candidates.append([])
                continue

            # select candidates for best position according
            # to calculated score (current token will be always at position 0)
            sort_value_indices = sorted(
                enumerate(sentence_candidates_scores_pairs[best_position]),
                key=lambda x: x[1][1] - x[1][0], reverse=True
            )
            sort_indices = [x[0] for x in sort_value_indices]
            current_token_idx = sort_indices.index(
                current_tokens_candidates_indices[num_sent][best_position]
            )
            del sort_indices[current_token_idx]
            sort_indices.insert(
                0, current_tokens_candidates_indices[num_sent][best_position]
            )
            if num_selected_candidates is None:
                selected_sort_indices = sort_indices
            else:
                selected_sort_indices = sort_indices[:num_selected_candidates]
            positions_candidates.append(
                [
                    candidates[num_sent][best_position][idx]
                    for idx in selected_sort_indices
                ]
            )

        # correct best_positions according to initial_mappings
        best_positions_adjusted = []
        for best_position, indices_mapping in zip(best_positions,
                                                  indices_mappings):
            if best_position is None:
                best_positions_adjusted.append(None)
            else:
                best_positions_adjusted.append(indices_mapping[best_position])
        return (
            best_positions_adjusted,
            (positions_scores_cur, positions_scores_pred),
            positions_candidates
        )

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

    def _calculate_position_score(
            self, position_left: int, position_right: int,
            scores: Tuple[float, float]
    ) -> float:
        """Calculate the score of current position based
            on scores of language models.

        :param position_left: position from left border
        :param position_right: position from right border
        :param scores: position

        :returns: calculated score
        """
        # TODO: better algorithm of scoring
        left_right_score, right_left_score = scores
        # make scores positive (for interpretability)
        result_score = 2 * (
                (left_right_score * right_left_score)
                / (left_right_score + right_left_score)
        )
        return result_score