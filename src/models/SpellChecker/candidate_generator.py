import re
from typing import List, Tuple, Iterable, Dict, Any
from string import punctuation

from deeppavlov.models.spelling_correction.levenshtein import (
    LevenshteinSearcherComponent
)


class LevenshteinSearcher:
    """Class-wrapper for making search of similar words based
    on Damerau-Levenshtein distance.
    """

    def __init__(self, words: Iterable[str], max_distance: int = 1):
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
        # get rid of scores and ignore initial token
        candidates = [
            [
                [candidate[1] for candidate in candidates_position
                 if candidate[1] != batch[i][j]]
                for j, candidates_position in enumerate(candidates_sentence)
            ]
            for i, candidates_sentence in enumerate(candidates_with_probs)
        ]
        return candidates


class HandcodeSearcher:
    """Class for manually adding candidates."""

    def __init__(self, table: Dict[str, List[str]]):
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
            return answer
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

    def __init__(self, words: Iterable[str]):
        """Init object.

        :param words: possible words to handle
        """
        # normalize words
        words = {
            re.sub(f'[{punctuation}]', '',
                   word.strip().lower().replace('ё', 'е'))
            for word in words if re.fullmatch(f'[{self.alphabet}-]+', word)
        }

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

        :returns: phonetic code.
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
        self.words = {
            word.strip().lower().replace('ё', 'е') for word in words
        }
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
            self, tokenized_sentences_cased: List[List[str]]
    ) -> List[List[List[Dict[str, Any]]]]:
        """Create candidates and their features
        for each position in each sentence.

        :param tokenized_sentences_cased: list of tokenized sentences
            before making lowercase

        :returns: list of candidates and their feautures
        for each sentence for each position
        """
        # make lowercase
        tokenized_sentences = [
            [x.lower() for x in sentence]
            for sentence in tokenized_sentences_cased
        ]

        # collect information about position
        positional_features = [
            [{'is_title': token.istitle(),
              'is_upper': token.isupper(),
              'is_lower': token.islower(),
              'is_first': j == 0
              } for j, token in enumerate(sentence)]
            for i, sentence in enumerate(tokenized_sentences_cased)
        ]

        candidates_levenshtein = self.levenshtein_searcher(tokenized_sentences)
        candidates_phonetic = self.phonetic_searcher(tokenized_sentences)
        candidates_handcode = self.handcode_searcher(tokenized_sentences)

        # collect all candidates and their features
        candidates: List[List[List[Dict[str, Any]]]] = []
        for i in range(len(tokenized_sentences)):
            candidates_sentence: List[List[Tuple[str, Dict[str, Any]]]] = []
            for j in range(len(tokenized_sentences[i])):
                # make dict not to duplicate candidates
                candidates_position: Dict[str, Dict[str, Any]] = {}
                # add initial token
                candidate = tokenized_sentences[i][j]
                candidates_position[candidate] = {}
                candidates_position[candidate].update(
                    self._calculate_features(candidate,
                                             positional_features[i][j])
                )
                # only initial token can be non-vocabulary word
                candidates_position[candidate].update({
                    'from_vocabulary': candidate in self.words,
                    'is_original': True,
                    'is_current': True
                })

                # add candidates from levenshtein searcher
                self._add_candidates_from_list(
                    candidates_position,
                    candidates_levenshtein[i][j],
                    positional_features[i][j],
                    {'from_levenshtein_searcher': True}
                )
                # add candidates from phonetic searcher
                self._add_candidates_from_list(
                    candidates_position,
                    candidates_phonetic[i][j],
                    positional_features[i][j],
                    {'from_phonetic_searcher': True}
                )
                # add candidates from handcode searcher
                self._add_candidates_from_list(
                    candidates_position,
                    candidates_handcode[i][j],
                    positional_features[i][j],
                    {'from_handcode_searcher': True}
                )

                candidates_sentence.append(
                    list(candidates_position.values())
                )
            candidates.append(candidates_sentence)

        return candidates

    def _calculate_features(
            self, token: str, positional_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculation of features for candidate.

        :param token: token value
        :param positional_features: information about position

        :returns: dictionary with features
        """
        features = {}
        # add candidate itself
        features['token'] = token
        # add positional features
        features.update(positional_features)
        # add feature about space/hyphen
        features['contains_space'] = (' ' in token)
        features['contains_hyphen'] = ('-' in token)
        # default features
        features['from_levenshtein_searcher'] = False
        features['from_phonetic_searcher'] = False
        features['from_handcode_searcher'] = False
        features['is_combined'] = False
        features['is_original'] = False
        features['is_current'] = False
        features['from_vocabulary'] = True
        return features

    def _add_candidates_from_list(
            self, candidates_with_features: Dict[str, Dict[str, Any]],
            candidates_raw: List[str],
            positional_features: Dict[str, Any],
            update_dict: Dict[str, Any]
    ) -> None:
        """Add candidates with features from list of candidate tokens.

        :param candidates_with_features:  list of candidates with features
        :param candidates_raw: list of candidates tokens to add
        :param positional_features: information about position
        :param update_dict: additional features to add
        """
        for candidate in candidates_raw:
            if candidate not in candidates_with_features:
                candidates_with_features[candidate] = (
                    self._calculate_features(
                        candidate, positional_features
                    )
                )
            candidates_with_features[candidate].update(update_dict)

    def combine_tokens(
            self, candidates: List[List[List[Dict[str, Any]]]]
    ) -> Tuple[List[List[List[Dict[str, Any]]]], List[List[int]]]:
        """Combine some consecutive tokens in one token
        to process more subtle cases.

        :param candidates: list of candidates and their features

        :returns:
            candidates after combination some tokens
            indices for combination tokens
        """

        candidates_combined = []
        indices_combined = []
        # process each sentence separately
        for num_sent, candidates_sentence in enumerate(candidates):
            # for combination we need at least len == 1
            if len(candidates_sentence) == 0:
                candidates_combined.append([])
                indices_combined.append([])
                continue

            # prepare arrays for combination
            indices_sentence_combined = [[0]]
            candidates_sentence_combined = [candidates_sentence[0]]

            # check all consecutive pair of tokens using already combined
            for i in range(len(candidates_sentence) - 1):
                # take first token as token of last combined group
                token_first = candidates_sentence_combined[-1][0]['token']
                # take second token as new token to combine
                token_second = candidates_sentence[i+1][0]['token']
                # try to remove space or replace it with hyphen
                combined_token = f'{token_first} {token_second}'
                candidates_combination = []
                space_token = f'{token_first}{token_second}'
                hyphen_token = f'{token_first}-{token_second}'

                # check that one of token is not punctuation
                if not (
                        re.fullmatch('[' + punctuation + ']+', token_first)
                        or re.fullmatch('[' + punctuation + ']+', token_second)
                ):
                    if space_token in self.words:
                        candidates_combination.append(space_token)
                    if hyphen_token in self.words:
                        candidates_combination.append(hyphen_token)

                # make combination
                if len(candidates_combination) > 0:
                    # make dict not to duplicate candidates
                    candidates_position: Dict[str, Dict[str, Any]] = {}

                    candidates_first = candidates_sentence_combined[-1]
                    candidates_second = candidates_sentence[i+1]
                    # new list of candidates consists of
                    # 1. original combined token
                    # 2. candidates for combined token
                    # 3. original first token + candidates for second token
                    # 4. candidates for first token + original second token

                    # 1. original combined token
                    original_candidate = combined_token
                    candidates_position[original_candidate] = {}
                    # calculate positional features according to some logic
                    positional_features = {}
                    positional_features['is_first'] = (
                        candidates_first[0]['is_first']
                    )
                    positional_features['is_lower'] = (
                        candidates_first[0]['is_lower']
                        and candidates_second[0]['is_lower']
                    )
                    positional_features['is_upper'] = (
                        candidates_first[0]['is_upper']
                        and candidates_second[0]['is_upper']
                    )
                    positional_features['is_title'] = (
                        candidates_first[0]['is_title']
                    )
                    candidates_position[original_candidate].update(
                        self._calculate_features(
                            combined_token, positional_features
                        )
                    )
                    candidates_position[original_candidate].update({
                        'from_vocabulary': all(
                            [token in self.words for token in combined_token]
                        ),
                        'is_original': True,
                        'is_current': True
                    })

                    # 2. candidates for combined token
                    # take positional features from first position
                    for token in candidates_combination:
                        candidates_position[token] = self._calculate_features(
                            token, positional_features
                        )
                        candidates_position[token]['is_combined'] = True

                    # 3. original first token + candidates for second token
                    for candidate_second in candidates_second[1:]:
                        candidate_first = candidates_first[0]
                        token_first = candidate_first['token']
                        token_second = candidate_second['token']
                        token = f"{token_first} {token_second}"
                        if token not in candidates_position:
                            candidates_position[token] = {}
                        candidates_position[token] = candidate_second
                        candidates_position[token].update(positional_features)
                        candidates_position[token]['token'] = token

                    # 4. candidates for first token + original second token
                    for candidate_first in candidates_first[1:]:
                        candidate_second = candidates_second[0]
                        token_first = candidate_first['token']
                        token_second = candidate_second['token']
                        token = f"{token_first} {token_second}"
                        if token not in candidates_position:
                            candidates_position[token] = {}
                        candidates_position[token] = candidate_first
                        candidates_position[token].update(positional_features)
                        candidates_position[token]['token'] = token

                    # add index for current group of combination
                    indices_sentence_combined[-1].append(i+1)
                    # update current combination group
                    candidates_sentence_combined[-1] = list(
                        candidates_position.values()
                    )

                # finish group of combination
                else:
                    # finish current group of combination
                    indices_sentence_combined.append([i+1])
                    # make new group of combination
                    candidates_sentence_combined.append(
                        candidates_sentence[i + 1]
                    )

            candidates_combined.append(candidates_sentence_combined)
            indices_combined.append(indices_sentence_combined)

        return candidates_combined, indices_combined
