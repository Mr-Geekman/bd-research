from typing import List, Callable


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
            preprocessor: Callable,
            num_selected_candidates: int = 64,
            max_it: int = 10,
    ):
        self.candidate_generator = candidate_generator
        self.position_selector = position_selector
        self.candidate_scorer = candidate_scorer
        self.stopping_criteria = stopping_criteria
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.preprocessor = preprocessor
        self.num_selected_candidates = num_selected_candidates
        self.max_it = max_it

    def __call__(self, sentences: List[str]) -> List[str]:
        """Make corrections for sentences.

        :param sentences: list of sentences

        :returns: list of corrected sentences
        """
        # make tokenization
        tokenized_sentences = self.tokenizer(sentences)

        # make preprocessing on tokens (e.g. remove punctuation, lowercase)
        tokenized_sentences = self.preprocessor(tokenized_sentences)

        # find correction for each token and their scores
        candidates = self.candidate_generator(tokenized_sentences)

        # add original tokens to candidates with zero probability
        for i in range(len(candidates)):
            for j in range(len(candidates[i])):
                current_candidates = [
                    x[1] for x in candidates[i][j]
                ]
                if tokenized_sentences[i][j] not in current_candidates:
                    candidates[i][j].append((0, tokenized_sentences[i][j]))

        # list of results
        corrected_sentences = [[] for _ in range(len(tokenized_sentences))]
        # indices of processed sentences
        indices_processed_sentences = list(range(len(tokenized_sentences)))

        # start iteration procedure
        for cur_it in range(self.max_it):

            # find indices of current tokens in lists of candidates
            current_tokens_candidates_indices = []
            for i in range(len(tokenized_sentences)):
                sentence_current_tokens_candidates_indices = []
                for j in range(len(tokenized_sentences[i])):
                    current_token = tokenized_sentences[i][j]
                    current_candidates = candidates[i][j]
                    sentence_current_tokens_candidates_indices.append(
                        current_candidates.index(current_token)
                    )
                current_tokens_candidates_indices.append(
                    sentence_current_tokens_candidates_indices
                )

            # find the best positions for corrections
            best_positions, positions_candidates = self.position_selector(
                tokenized_sentences, candidates,
                current_tokens_candidates_indices,
                self.num_selected_candidates
            )

            # make scoring of candidates
            scoring_results = self.candidate_scorer(
                tokenized_sentences, best_positions,
                positions_candidates, self.detokenizer
            )

            # make best corrections
            current_scores = [
                scoring_results[idx][current_tokens_candidates_indices[idx][
                    best_positions[idx]]
                ]
                for idx in range(len(scoring_results))
            ]
            best_scores_with_indices = [
                max(enumerate(sentence_scoring_results), key=lambda x: x[1])
                for sentence_scoring_results in scoring_results
            ]
            best_scores = [x[1] for x in best_scores_with_indices]
            best_candidates = [
                positions_candidates[x[0]] for x in best_scores_with_indices
            ]
            for i in range(len(tokenized_sentences)):
                tokenized_sentences[i][best_positions[i]] = best_candidates[i]

            # check stopping criteria and finish some sentences
            criteria_results = self.stopping_criteria(
                current_scores, best_scores
            )
            for i in range(len(tokenized_sentences)):
                if criteria_results[i]:
                    idx = indices_processed_sentences[i]
                    corrected_sentences[idx] = tokenized_sentences[i]
                    del tokenized_sentences[i]
                    del indices_processed_sentences[i]

        # process remain sentences if they aren't finished
        for i in range(len(tokenized_sentences)):
            idx = indices_processed_sentences[i]
            corrected_sentences[idx] = tokenized_sentences[i]

        # return current detokenized sentences
        return self.detokenizer(corrected_sentences)
