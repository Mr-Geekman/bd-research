from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import kenlm

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.component import Component

logger = getLogger(__name__)


class KenlmPruner(Component):
    """
    Component that chooses some a candidates with the highest products
        of base and language model probabilities.
    """

    def __init__(self, load_path: Path, beam_size: int = 100, *args, **kwargs):
        """Init object.
        
        :param load_path: path to the kenlm model file
        :param beam_size: beam size for highest probability search
        """
        self.lm = kenlm.Model(str(expand_path(load_path)))
        self.beam_size = beam_size

    def __call__(self, batch: List[List[List[Tuple[float, str]]]]) -> List[List[List[Tuple[float, str]]]]:
        """Choose beam_size best hypotheses.

        :param batch: batch of probabilities and string values of candidates
            for every token in a sentence

        :returns: batch of corrected tokenized sentences
        """
        return [self._infer_instance(candidates) for candidates in batch]

    def _infer_instance(self, candidates: List[List[Tuple[float, str]]]):
        candidates = candidates + [[(0, '</s>')]]
        state = kenlm.State()
        self.lm.BeginSentenceWrite(state)
        beam = [(0, state, [])]
        for sublist in candidates:
            new_beam = []
            for beam_score, beam_state, beam_words in beam:
                for score, candidate in sublist:
                    prev_state = beam_state
                    c_score = 0
                    cs = candidate.split()
                    for candidate in cs:
                        state = kenlm.State()
                        c_score += self.lm.BaseScore(prev_state, candidate, state)
                        prev_state = state
                    new_beam.append((beam_score + score + c_score, state, beam_words + cs))
            new_beam.sort(reverse=True)
            beam = new_beam[:self.beam_size]
        return list([(score, words) for (score, state, words) in beam])
