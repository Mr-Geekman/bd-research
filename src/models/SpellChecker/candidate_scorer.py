from typing import List, Tuple, Dict, Any, Optional


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
            candidates_all: List[List[List[Dict[str, Any]]]],
            return_scoring_info: bool = False
    ) -> Tuple[List[int], List[List[float]], Optional[Any]]:
        """Make scoring for candidates for every sentence and adjust them.

        :param tokenized_sentences: list of tokenized sentences
        :param positions: positions for candidates scoring for each sentence
        :param candidates_all: all candidates and their features
            for each positions in each sentence
        :param return_scoring_info: return or not return additional
            information about scoring

        :returns:
            indices of best candidates for each position
            results of scoring
            additional information about scoring
        """
        # select candidates by their positions
        candidates = [candidates_all[num_sent][pos]
                      for num_sent, pos in enumerate(positions)]

        # make scoring
        scoring_results, scoring_info = self.scorer(
            tokenized_sentences, positions, candidates,
            return_scoring_info=return_scoring_info
        )

        # find best corrections
        best_scores_with_indices = [
            max(enumerate(sentence_scoring_results), key=lambda x: x[1])
            for sentence_scoring_results in scoring_results
        ]
        best_indices = [x[0] for x in best_scores_with_indices]

        return best_indices, scoring_results, scoring_info
