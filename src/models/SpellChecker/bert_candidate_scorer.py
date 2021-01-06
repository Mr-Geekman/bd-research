from typing import List, Callable
from copy import copy

from src.models.BertScorer.bert_scorer_correction import BertScorerCorrection


class BertScorerWrapper:
    """Wrapper class over BertScorerCorrection
    to score candidates for correction.
    """

    def __init__(
            self,
            bert_scorer_model: BertScorerCorrection,
    ):
        self.bert_scorer_model = bert_scorer_model

    def __call__(
            self, tokenized_sentences: List[List[str]], positions: List[int],
            candidates: List[List[str]], detokenizer: Callable
    ) -> List[List[float]]:
        """Make scoring for candidates for every sentence.

        :param tokenized_sentences: list of tokenized sentences
        :param positions: positions for candidates scoring for each sentence
        :param candidates: candidates for given positions
            in each sentence
        :param detokenizer: object to make sentence
            string from tokenized sentence

        :returns: score for each candidate for each sentence
        """
        # add mask tokens to given positions
        masked_tokenized_sentences = []
        for i, pos in enumerate(positions):
            current_sentence = copy(tokenized_sentences[i])
            current_sentence[pos] = self.bert_scorer_model.tokenizer.mask_token
            masked_tokenized_sentences.append(current_sentence)

        # detokenize sentences
        masked_sentences = detokenizer(masked_tokenized_sentences)

        # make scoring
        scoring_results = self.bert_scorer_model(masked_sentences, candidates)

        return scoring_results
