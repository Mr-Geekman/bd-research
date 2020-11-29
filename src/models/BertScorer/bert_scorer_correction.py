from typing import List, Dict, Callable
from copy import copy

import numpy as np
import torch
from transformers.tokenization_utils import PreTrainedTokenizer, BatchEncoding
from transformers import BertForMaskedLM


class BertScorerCorrection:
    """
    Class for scoring all candidates for correction by BERT model.
    """

    def __init__(
            self,
            model: BertForMaskedLM,
            tokenizer: PreTrainedTokenizer,
            device: int = -1
    ):
        self.device = torch.device('cpu' if device < 0 else f'cuda:{device}')
        self.model = model.to(device=self.device)
        self.tokenizer = tokenizer

    def __call__(self, sentence: str, candidates: List[str],
                 agg_func: Callable = np.mean) -> Dict[str, float]:
        """
        Make scoring for candidates.

        :param sentence: sentence with mask token, indicates, where we should
            place candidates
        :param candidates: list of candidates to score
        :param agg_func: function, that aggregates scores
            for multi-token candidate

        :returns: dict with candidate and its score
        """
        # check, that there is mask token in sentence
        tokenized_sentence = self.tokenizer(
            sentence,
            add_special_tokens=True,
            padding=True,
            truncation='only_first',
        )
        if tokenized_sentence[
            'input_ids'
        ].count(self.tokenizer.mask_token_id) != 1:
            raise ValueError(
                'Where should be exactly one [MASK] token in a sentence.'
            )

        # find mask index
        mask_index = tokenized_sentence[
            'input_ids'
        ].index(self.tokenizer.mask_token_id)

        # tokenize all candidates
        tokenized_candidates = self.tokenizer(
            candidates,
            add_special_tokens=False,
            padding=False,
            truncation='do_not_truncate',
        )

        # score each candidate
        candidates_scores = {}
        for candidate, tokenized_candidate in zip(
                candidates, tokenized_candidates['input_ids']
        ):
            candidates_scores[candidate] = self._score_candidate(
                    tokenized_sentence, mask_index,
                    tokenized_candidate, agg_func
            )
        return candidates_scores

    def _score_candidate(
            self, tokenized_sentence: BatchEncoding,
            mask_index: int, tokenized_candidate: List[int],
            agg_func: Callable
    ) -> float:
        """
        Scoring of single candidate.

        :param tokenized_sentence: source sentence with [MASK] token
        :param mask_index: index of [MASK] token
        :param tokenized_candidate: input_ids of candidate
        :param agg_func: function, that aggregates scores
            for multi-token candidate

        :returns: result of scoring
        """
        # we make scoring with [MASK] token in each possible position in
        # candidate and take mean value of scores
        scores = []
        for i in range(len(tokenized_candidate)):
            # make substitution for [MASK] token in tokenized sentence
            mask_substitution = copy(tokenized_candidate)
            mask_substitution[i] = self.tokenizer.mask_token_id

            # create valid BatchEncoding object to make scoring
            input_dict = {'input_ids': (
                torch.tensor(
                    tokenized_sentence['input_ids'][:mask_index]
                    + mask_substitution
                    + tokenized_sentence['input_ids'][mask_index + 1:],
                    dtype=torch.long, device=self.device
                ).unsqueeze(0)
            )}
            input_dict['attention_mask'] = torch.ones(
                len(input_dict['input_ids']),
                dtype=torch.long, device=self.device
            ).unsqueeze(0)
            input_dict['token_type_ids'] = torch.zeros(
                len(input_dict['input_ids']),
                dtype=torch.long, device=self.device
            ).unsqueeze(0)
            model_input = BatchEncoding(input_dict)
            with torch.no_grad():
                model_output = self.model(**model_input)[0].squeeze(0).cpu()
                log_probs = torch.nn.functional.log_softmax(
                    model_output[i, :], dim=0
                )
                scores.append(log_probs[tokenized_candidate[i]].item())
        return agg_func(scores)
