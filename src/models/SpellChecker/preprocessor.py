import re
from typing import List
from string import punctuation


class Preprocessor:
    """Preprocessor for my specific task, that removes punctuation."""

    def __init__(
            self
    ):
        pass

    def __call__(self,
                 tokenized_sentences: List[List[str]]) -> List[List[str]]:
        """Make preprocessing.

        :param tokenized_sentences: list of tokenized sentences

        :returns: list of processed tokenized sentences
        """
        # make lowercase
        lowered_tokenized_sentences = [
            [x.lower() for x in tokenized_sentence]
            for tokenized_sentence in tokenized_sentences
        ]

        # delete punctuation
        processed_sentences = [
            [x for x in tokenized_sentence
             if not re.fullmatch('[' + punctuation + ']+', x)]
            for tokenized_sentence in lowered_tokenized_sentences
        ]

        return processed_sentences
