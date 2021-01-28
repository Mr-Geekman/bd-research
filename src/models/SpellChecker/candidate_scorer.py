from typing import List, Tuple, Dict, Any


class CandidateScorer:
    """Class that using features  to score candidates for correction."""

    def __init__(self, scorer):
        """Init object.

        :param scorer: model for scoring based of features of candidates
        """
        self.scorer = scorer

    def __call__(
            self, tokenized_sentences: List[List[str]],
            positions: List[int],
            candidates_features: List[List[Tuple[str, Dict[str, Any]]]]
    ) -> Tuple[List[int], Tuple[List[float], List[float]], List[List[float]]]:
        """Make scoring for candidates for every sentence and adjust them.

        :param tokenized_sentences: list of tokenized sentences
        :param positions: positions for candidates scoring for each sentence
        :param candidates_features: candidates and their features
            for given positions in each sentence

        :returns:
            indices of best candidates for each position
            (list of current scores for each sentence,
            list of best scores for each sentence)
            results of scoring
        """
        # make scoring
        scoring_results = self.scorer(
            tokenized_sentences, positions, candidates_features
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

        return best_indices, (current_scores, best_scores), scoring_results
