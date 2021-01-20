import re
from typing import List, Callable, Any
from string import punctuation


def bool_anti_index(
        bool_list: List[bool],
        list_to_process: List[Any],
) -> List[Any]:
    """Indexing of list according to criteria results.
        Pass the element to new list iff result is False

    :param bool_list: list of bool values
    :param list_to_process: list to process

    :returns: processed list
    """
    processed_list = [
        list_to_process[i]
        for i, res in enumerate(bool_list) if not res
    ]
    return processed_list


class IterativeSpellChecker:
    """Class that makes spell checking with iterative refinement."""

    def __init__(
            self,
            candidate_generator: Callable,
            position_selector: Callable,
            candidate_scorer: Callable,
            position_stopping_criteria: Callable,
            scorer_stopping_criteria: Callable,
            tokenizer: Callable,
            detokenizer: Callable,
            num_selected_candidates: int = 16,
            max_it: int = 5,
    ):
        self.candidate_generator = candidate_generator
        self.position_selector = position_selector
        self.candidate_scorer = candidate_scorer
        self.position_stopping_criteria = position_stopping_criteria
        self.scorer_stopping_criteria = scorer_stopping_criteria
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.num_selected_candidates = num_selected_candidates
        self.max_it = max_it

    def __call__(self, sentences: List[str]) -> List[str]:
        """Make corrections for sentences.

        :param sentences: list of sentences

        :returns: list of corrected sentences
        """
        # make tokenization
        tokenized_sentences_start = [
            self.tokenizer(sentence) for sentence in sentences
        ]
        # make lowercase
        tokenized_sentences = [
            [x.lower().replace('ё', 'е') for x in sentence]
            for sentence in tokenized_sentences_start
        ]

        # find correction for each token and their scores
        candidates = self.candidate_generator(tokenized_sentences)

        # list of results
        corrected_sentences = [[] for _ in range(len(tokenized_sentences))]
        # indices of processed sentences
        indices_processing_sentences = list(range(len(tokenized_sentences)))

        # start iteration procedure
        for cur_it in range(self.max_it):

            # find indices of current tokens in lists of candidates
            # TODO: to avoid this
            current_tokens_candidates_indices_all_positions = []
            for i in range(len(tokenized_sentences)):
                sentence_current_tokens_candidates_indices = []
                for j in range(len(tokenized_sentences[i])):
                    current_token = tokenized_sentences[i][j]
                    current_candidates = [x[1] for x in candidates[i][j]]
                    sentence_current_tokens_candidates_indices.append(
                        current_candidates.index(current_token)
                    )
                current_tokens_candidates_indices_all_positions.append(
                    sentence_current_tokens_candidates_indices
                )

            # find the best positions for corrections
            # TODO: think about more flexible finding positions
            #   e.g. we can somehow track other good positions and use them
            #   then best_position was unsuccessfully
            #   used on previous iteration
            position_selector_results = self.position_selector(
                tokenized_sentences, candidates,
                current_tokens_candidates_indices_all_positions,
                self.num_selected_candidates
            )
            best_positions = position_selector_results[0]
            positions_scores = position_selector_results[1]
            positions_candidates = position_selector_results[2]

            # check stopping criteria for position selector
            # and finish some sentences
            criteria_results = self.position_stopping_criteria(
                positions_scores[0], positions_scores[1]
            )
            self._finish_sentences(
                criteria_results, tokenized_sentences,
                indices_processing_sentences, corrected_sentences
            )
            indices_processing_sentences = bool_anti_index(
                criteria_results, indices_processing_sentences
            )
            tokenized_sentences = bool_anti_index(criteria_results,
                                                  tokenized_sentences)
            candidates = bool_anti_index(criteria_results, candidates)
            best_positions = bool_anti_index(criteria_results, best_positions)
            positions_candidates = bool_anti_index(criteria_results,
                                                   positions_candidates)
            # if all sentences was processed before reaching max_it
            if len(tokenized_sentences) == 0:
                break

            # make scoring of candidates
            best_candidates, candidate_scores = self.candidate_scorer(
                tokenized_sentences, best_positions,
                positions_candidates
            )

            # make best corrections
            for i in range(len(tokenized_sentences)):
                tokenized_sentences[i][best_positions[i]] = best_candidates[i]

            # check stopping criteria for candidate scorer
            # and finish some sentences
            criteria_results = self.scorer_stopping_criteria(
                candidate_scores[0], candidate_scores[1]
            )
            self._finish_sentences(
                criteria_results, tokenized_sentences,
                indices_processing_sentences, corrected_sentences
            )
            indices_processing_sentences = bool_anti_index(
                criteria_results, indices_processing_sentences
            )
            candidates = bool_anti_index(criteria_results, candidates)
            tokenized_sentences = bool_anti_index(criteria_results,
                                                  tokenized_sentences)
            # if all sentences was processed before reaching max_it
            if len(tokenized_sentences) == 0:
                break

        # process remain sentences if they aren't finished
        for i in range(len(tokenized_sentences)):
            idx = indices_processing_sentences[i]
            corrected_sentences[idx] = tokenized_sentences[i]

        # remove punctuation from sentences
        corrected_sentences = [
            [x for x in sentence
             if not re.fullmatch('[' + punctuation + ']+', x)]
            for sentence in corrected_sentences
        ]

        # return current detokenized sentences
        return [
            self.detokenizer(sentence) for sentence in corrected_sentences
        ]

    def _finish_sentences(
            self, criteria_results: List[bool],
            tokenized_sentences: List[List[str]],
            indices_processing_sentences: List[int],
            corrected_sentences: List[List[str]]
    ) -> None:
        """Finish sentences according to result of stopping criteria.

        :param criteria_results: results of stopping criteria
        :param tokenized_sentences: list of tokenized sentences
        :param indices_processing_sentences: indices  of current
            processing sentences in initial list of processing sentences
        :param corrected_sentences: finished tokenized sentences
            (it is updated in this method)
        """
        for i in range(len(tokenized_sentences)):
            idx = indices_processing_sentences[i]
            if criteria_results[i]:
                corrected_sentences[idx] = tokenized_sentences[i]
