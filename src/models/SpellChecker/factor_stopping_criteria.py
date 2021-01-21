from typing import List, Optional


class FactorStoppingCriteria:
    """Stopping criteria according to factor between
    current results and previous results.
    """

    def __init__(
            self,
            factor_constant,
    ):
        """Init object.

        :param factor_constant: constant for comparison
        """
        self.factor_constant = factor_constant

    def __call__(
            self, current_scores: List[float],
            new_scores: List[Optional[float]]
    ) -> List[bool]:
        """Check for each sentence criteria of continuation of correction.

        :param current_scores: list of scores for current tokens
            in selected position in each sentence
        :param new_scores: list of scores for best tokens
            in selected positions in each sentence

        :returns: list of indicators should we stop or not
        """
        results = [
            new_score is None
            or (new_score/current_score <= self.factor_constant)
            for current_score, new_score in zip(current_scores, new_scores)
        ]
        return results
