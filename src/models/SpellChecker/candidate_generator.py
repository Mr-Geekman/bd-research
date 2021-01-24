import re
from typing import List, Iterable, Dict
from string import punctuation

from deeppavlov.models.spelling_correction.levenshtein import (
    LevenshteinSearcherComponent
)


class LevenshteinSearcher:
    """Class-wrapper for making search of similar words based
    on Damerau-Levenshtein distance.
    """

    def __init__(self, words: List[str], max_distance: int = 1):
        """Init object.

        :param words: possible words of candidates
        :param max_distance: max distance for LevenshteinSearcherComponent
        """
        self.model = LevenshteinSearcherComponent(
            words=words, max_distance=max_distance
        )

    def __call__(self, batch: List[List[str]]) -> List[List[List[str]]]:
        """Propose candidates for tokens in sentences

        :param batch: batch of tokenized sentences

        :returns: list of candidates for each position
            for each sentence of batch
        """
        candidates_with_probs = self.model(batch)
        # get rid of scores
        candidates = [
            [
                [candidate[1] for candidate in candidates_position]
                for candidates_position in candidates_sentence
            ]
            for candidates_sentence in candidates_with_probs
        ]
        return candidates


class HandcodeSearcher:
    """Class for manually adding candidates."""

    def __init__(self, table: Dict[str, str]):
        """Init object.

        :param table: table of handcoded candidates
        """
        self.table = table

    def _infer_token(self, token: str) -> List[str]:
        """Generate candidates for one token.

        :param token: one token

        :returns: list of candidates
        """
        # normalize token
        token = re.sub(f'[{punctuation}]', '', token.lower().replace('ё', 'е'))
        # make query in table
        answer = self.table.get(token)
        if answer:
            return [answer]
        else:
            return []

    def __call__(self, batch: List[List[str]]) -> List[List[List[str]]]:
        """Propose candidates for tokens in sentences

        :param batch: batch of tokenized sentences

        :returns: list of candidates for each position
            for each sentence of batch
        """
        return [[self._infer_token(token) for token in sentence]
                for sentence in batch]


class PhoneticSeacher:
    """Class for making phonetic search of similar words."""
    alphabet = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
    sibilants = 'сзшшч'
    consonants = 'бвгджзйклмнпрстфхцчшщ'
    vowels = 'аиоуыэеюя'
    signs = 'ьъ'
    inv_table = {
        1: ['а', 'о', 'ы', 'у', 'я'],
        3: ['и', 'е', 'ю', 'я', 'э'],
        5: ['б', 'п'],
        6: ['в', 'ф'],
        7: ['д', 'т'],
        8: ['г', 'к', 'х'],
        9: ['л'],
        10: ['р'],
        11: ['м'],
        12: ['н'],
        13: ['з', 'с'],
        14: ['й'],
        15: ['щ', 'ч'],
        16: ['ж', 'ш'],
        17: ['ц']
    }

    def __init__(self, words: List[str]):
        """Init object.

        :param words: possible words to handle
        """
        # normalize words
        words = list({
            re.sub(f'[{punctuation}]', '',
                   word.strip().lower().replace('ё', 'е'))
            for word in words if ' ' not in word
        })

        # create mapping table
        self.table = {}
        for key, values in self.inv_table.items():
            for value in values:
                self.table[value] = key

        # make mapping for all dictionary words
        self.words_mapping = {}
        for word in words:
            code = tuple(self._make_phonetic_encoding(word))
            if code not in self.words_mapping:
                self.words_mapping[code] = []
            self.words_mapping[code].append(word)

    def _make_phonetic_encoding(self, word: str) -> List[int]:
        """Make phonetic encoding of the word.

        :param word: word to encode

        :return: phonetic code.
        """
        # make substitution for "тс", "тьс", "тъс"
        word = word.replace('тс', 'ц').replace('тьс', 'ц').replace('тъс', 'ц')
        # remove "т" after sibilants and before consonants
        word = re.sub(
            f'([{self.sibilants}])т([{self.vowels}])',
            '\\1\\2',
            word
        )
        # after "ьъщчй" make vowels class 3
        word = re.sub(
            f'([ьъщчй])[{self.vowels}]',
            '\\1е',
            word
        )
        # after "шжц" make vowels class 1
        word = re.sub(
            f'([шжц])[{self.vowels}]',
            '\\1а',
            word
        )
        # remove signs
        word = re.sub(
            f'[{self.signs}]',
            '',
            word
        )
        # make mapping according to table
        word_list = [
            self.table[char] if char not in ['1', '3'] else int(char)
            for char in word
        ]
        # remove repeating codes
        if word_list:
            cur_char = word_list[0]
            new_word_list = [word_list[0]]
            for char in word_list[1:]:
                if char != cur_char:
                    new_word_list.append(char)
                    cur_char = char
        else:
            return [0]
        return new_word_list

    def _infer_token(self, token: str) -> List[str]:
        """Generate candidates for one token.

        :param token: one token

        :returns: list of candidates
        """
        # normalize token
        token = re.sub(f'[{punctuation}]', '', token.lower().replace('ё', 'е'))

        if not re.fullmatch(f'[{self.alphabet}]+', token):
            return []
        else:
            code = tuple(self._make_phonetic_encoding(token))
            return self.words_mapping.get(code, [])

    def __call__(self, batch: List[List[str]]) -> List[List[List[str]]]:
        """Propose candidates for tokens in sentences

        :param batch: batch of tokenized sentences

        :returns: list of candidates for each position
            for each sentence of batch
        """
        return [[self._infer_token(token) for token in sentence]
                for sentence in batch]


class CandidateGenerator:
    """Class for generating candidates for correction."""

    def __init__(
            self,
            words: Iterable[str],
            handcode_table: Dict[str, str],
            max_distance: int = 1,
    ):
        """Init object.

        :param words: possible words of candidates
        :param max_distance: max distance for LevenshteinSearcherComponent
        :param handcode_table: table of handcoded candidates
        """
        self.words = list(words)
        self.max_distance = max_distance
        self.handcode_table = handcode_table
        # create levenstein searcher
        self.levenshtein_searcher = LevenshteinSearcher(
            words=self.words, max_distance=self.max_distance
        )
        # create phonetic searcher
        self.phonetic_searcher = PhoneticSeacher(words=self.words)
        # create handcode searcher
        self.handcode_searcher = HandcodeSearcher(self.handcode_table)

    def __call__(
            self, tokenized_sentences: List[List[str]],
    ) -> List[List[List[str]]]:
        """Create candidates for each position in each sentence.

        :param tokenized_sentences: list of tokenized sentences

        :returns: list of candidates for each sentence for each position
        """
        candidates_levenshtein = self.levenshtein_searcher(tokenized_sentences)
        candidates_phonetic = self.phonetic_searcher(tokenized_sentences)
        candidates_handcoded = self.handcode_searcher(tokenized_sentences)

        # unite candidates of different searchers
        candidates = [
            [
                list(set(
                    candidates_levenshtein[i][j]
                    + candidates_phonetic[i][j]
                    + candidates_handcoded[i][j]
                    + [tokenized_sentences[i][j]]
                ))
                for j in range(len(tokenized_sentences[i]))
            ]
            for i in range(len(tokenized_sentences))
        ]
        return candidates
