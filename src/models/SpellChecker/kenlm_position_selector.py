from typing import List, Tuple, Callable

import numpy as np
import kenlm


class KenlmPositionSelector:
    """Class that selects best positions for correction."""

    def __init__(
            self, left_right_model: kenlm.Model, right_left_model: kenlm.Model,
            agg_subtoken_func: Callable = np.sum
    ):
        self.left_right_model = left_right_model
        self.right_left_model = right_left_model
        self.agg_subtoken_func = agg_subtoken_func

    def __call__(
            self, tokenized_sentences: List[List[str]],
            candidates: List[List[List[Tuple[float, str]]]],
            current_tokens_candidates_indices: List[List[int]],
            num_selected_candidates: int,
    ) -> Tuple[List[int], List[List[str]]]:
        """Select best positions for correction and candidates to score.

        :param tokenized_sentences: list of sentences, that are list of tokens
        :param candidates: list of possible corrections
            for each position in each sentence
        :param current_tokens_candidates_indices: list of indices
            for current tokens in sentences
        :param num_selected_candidates: max number of selected tokens
            for each position in sentence

        :returns:
            list of best positions for each sentence,
            selected candidates for each position for each sentence
        """
        best_positions = []
        positions_candidates = []
        # process each sentence separately
        for num_sent, tokenized_sentence in enumerate(tokenized_sentences):
            sentence_candidates_scores = [
                [] for _ in range(len(tokenized_sentence))
            ]
            # make left-to-right pass
            state = kenlm.State()
            self.left_right_model.BeginSentenceWrite(state)
            prev_state = state
            for pos, token in enumerate(tokenized_sentence):
                for score, candidate in candidates[num_sent][pos]:
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
                    ][1], state
                )
                prev_state = state

            # make right-to-left pass
            state = kenlm.State()
            self.right_left_model.BeginSentenceWrite(state)
            prev_state = state
            for inv_pos, token in enumerate(tokenized_sentence[::-1]):
                pos = len(tokenized_sentence) - 1 - inv_pos
                for i, (score, candidate) in enumerate(
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
                    ][1], state
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

            # compare score to the score of current token
            sentence_candidates_scores_factors = []
            for pos, scores in enumerate(sentence_candidates_agg_scores):
                sentence_position_candidates_scores_factors = []
                for j, score in enumerate(scores):
                    sentence_position_candidates_scores_factors.append(
                        score
                        /
                        scores[
                            current_tokens_candidates_indices[num_sent][pos]
                        ]
                    )
                sentence_candidates_scores_factors.append(
                    sentence_position_candidates_scores_factors
                )

            # select best position
            max_positions_factors = [
                max(factors) for factors in sentence_candidates_scores_factors
            ]
            best_position = max(
                enumerate(max_positions_factors), key=lambda x: x[1]
            )[0]
            best_positions.append(best_position)

            # select candidates for best position according
            # to calculated factor (current token is always at position 0)
            sort_value_indices = sorted(
                enumerate(sentence_candidates_scores_factors[best_position]),
                key=lambda x: x[1], reverse=True
            )
            sort_indices = [x[0] for x in sort_value_indices]
            current_token_idx = sort_indices.index(
                current_tokens_candidates_indices[num_sent][best_position]
            )
            del sort_indices[current_token_idx]
            sort_indices.insert(
                0, current_tokens_candidates_indices[num_sent][best_position]
            )
            positions_candidates.append(
                [
                    candidates[num_sent][best_position][idx][1]
                    for idx in sort_indices[:num_selected_candidates]
                ]
            )

        return best_positions, positions_candidates

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
        left_right_score = -1/left_right_score
        right_left_score = -1/right_left_score
        result_score = 2 * (
                (left_right_score * right_left_score)
                / (left_right_score + right_left_score)
        )
        return result_score
