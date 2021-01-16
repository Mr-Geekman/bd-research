import re
from typing import List, Callable
from string import punctuation


class IterativeSpellChecker:
    """Class that makes spell checking with iterative refinement."""

    def __init__(
            self,
            candidate_generator: Callable,
            position_selector: Callable,
            candidate_scorer: Callable,
            stopping_criteria: Callable,
            tokenizer: Callable,
            detokenizer: Callable,
            num_selected_candidates: int = 16,
            max_it: int = 5,
    ):
        self.candidate_generator = candidate_generator
        self.position_selector = position_selector
        self.candidate_scorer = candidate_scorer
        self.stopping_criteria = stopping_criteria
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
            [x.lower() for x in sentence]
            for sentence in tokenized_sentences_start
        ]

        # find correction for each token and their scores
        candidates = self.candidate_generator(tokenized_sentences)

        # list of results
        corrected_sentences = [[] for _ in range(len(tokenized_sentences))]
        # indices of processed sentences
        indices_processed_sentences = list(range(len(tokenized_sentences)))

        # start iteration procedure
        for cur_it in range(self.max_it):

            # find indices of current tokens in lists of candidates
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
            best_positions, positions_candidates = self.position_selector(
                tokenized_sentences, candidates,
                current_tokens_candidates_indices_all_positions,
                self.num_selected_candidates
            )

            # make scoring of candidates
            scoring_results = self.candidate_scorer(
                tokenized_sentences, best_positions,
                positions_candidates, self.detokenizer
            )

            # make best corrections
            current_scores = [
                scoring_results[idx][0]
                for idx in range(len(scoring_results))
            ]
            best_scores_with_indices = [
                max(enumerate(sentence_scoring_results), key=lambda x: x[1])
                for sentence_scoring_results in scoring_results
            ]
            best_scores = [x[1] for x in best_scores_with_indices]
            best_candidates = [
                positions_candidates[i][x[0]]
                for i, x in enumerate(best_scores_with_indices)
            ]
            for i in range(len(tokenized_sentences)):
                tokenized_sentences[i][best_positions[i]] = best_candidates[i]

            # check stopping criteria and finish some sentences
            criteria_results = self.stopping_criteria(
                current_scores, best_scores
            )
            new_tokenized_sentences = []
            new_indices_processed_sentences = []
            new_candidates = []
            for i in range(len(tokenized_sentences)):
                if criteria_results[i]:
                    idx = indices_processed_sentences[i]
                    corrected_sentences[idx] = tokenized_sentences[i]
                else:
                    new_tokenized_sentences.append(tokenized_sentences[i])
                    new_indices_processed_sentences.append(i)
                    new_candidates.append(candidates[i])
            tokenized_sentences = new_tokenized_sentences
            indices_processed_sentences = new_indices_processed_sentences
            candidates = new_candidates

            # if all sentences was processed before reaching max_it
            if len(tokenized_sentences) == 0:
                break

        # process remain sentences if they aren't finished
        for i in range(len(tokenized_sentences)):
            idx = indices_processed_sentences[i]
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
