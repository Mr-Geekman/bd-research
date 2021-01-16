from typing import List, Tuple

from deeppavlov.models.spelling_correction.levenshtein import (
    LevenshteinSearcherComponent
)


class LevenshteinCandidateGenerator:
    """Wrapper class over LevenshteinSearcherComponent
    to generate candidates for correction.
    """

    def __init__(
            self,
            levenshtein_candidate_generator: LevenshteinSearcherComponent,
    ):
        self.levenshtein_candidate_generator = levenshtein_candidate_generator

    def __call__(
            self, tokenized_sentences: List[List[str]],
    ) -> List[List[List[Tuple[float, str]]]]:
        """Create candidates for each position in each sentence.

        :param tokenized_sentences: list of tokenized sentences

        :returns: list of candidates for each sentence for each position
        """
        # find candidates
        candidates = self.levenshtein_candidate_generator(tokenized_sentences)

        # add original tokens to candidates with zero probability
        for i in range(len(candidates)):
            for j in range(len(candidates[i])):
                current_candidates = [
                    x[1] for x in candidates[i][j]
                ]
                if tokenized_sentences[i][j] not in current_candidates:
                    candidates[i][j].append((0, tokenized_sentences[i][j]))
        return candidates
