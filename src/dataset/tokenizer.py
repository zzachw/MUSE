import os
from typing import List

import numpy as np
import torch

from src.dataset.vocab import Vocabulary
from src.utils import processed_data_path, load_pickle


def to_index(sequence: List[str], vocab, prefix="", suffix=""):
    """ convert code to index (each timestamp contains one token) """
    prefix = [vocab(prefix)] if prefix else []
    suffix = [vocab(suffix)] if suffix else []
    sequence = prefix + [vocab(token) for token in sequence] + suffix
    sequence = torch.tensor(sequence)
    return sequence


def to_vector(sequence: List[List[str]], vocab, prefix="", suffix=""):
    """ convert code to multihot vector (each timestamp contains many tokens) """
    if prefix:
        sequence = [[prefix]] + sequence
    if suffix:
        sequence = sequence + [[suffix]]
    multihot_vector = torch.zeros(len(sequence), len(vocab))
    for i, tokens in enumerate(sequence):
        for token in tokens:
            multihot_vector[i, vocab(token)] = 1
    return multihot_vector


def read(file, dtype='float'):
    with open(file) as file:
        header = file.readline().split(' ')
        count = int(header[0])
        dim = int(header[1])
        matrix = np.empty((count, dim), dtype=dtype)
        for i in range(count):
            matrix[i] = np.fromstring(file.readline(), sep=' ', dtype=dtype)
    return matrix


class MIMIC4Tokenizer:
    def __init__(self):
        self.code_vocabs, self.code_vocabs_size, self.code_embeddings = self._load_code_vocabs()
        self.type_vocabs, self.type_vocabs_size = self._load_type_vocabs()
        self.age_vocabs, self.age_vocabs_size = self._load_age_vocabs()
        self.gender_vocabs, self.gender_vocabs_size = self._load_gender_vocabs()
        self.ethnicity_vocabs, self.ethnicity_vocabs_size = self._load_ethnicity_vocabs()

    def _load_code_vocabs(self):
        vocab_dir = os.path.join(processed_data_path, f"mimic4/vocab.pkl")
        vocabs = load_pickle(vocab_dir)
        vocabs_size = len(vocabs)
        embeddings = read(os.path.join(processed_data_path, f"mimic4/embeddings.txt"))
        return vocabs, vocabs_size, embeddings

    def _load_type_vocabs(self):
        vocabs = Vocabulary()
        for word in [
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
        ]:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_age_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in range(18, 90):
            word = word // 10 * 10
            vocabs.add_word(str(word))
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_gender_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in ["F", "M"]:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_ethnicity_vocabs(self):
        all_ethnicities = [
            'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
            'HISPANIC OR LATINO',
            'BLACK/AFRICAN',
            'BLACK/CAPE VERDEAN',
            'BLACK/CARIBBEAN ISLAND',
            'HISPANIC/LATINO - HONDURAN',
            'WHITE - OTHER EUROPEAN',
            'WHITE',
            'HISPANIC/LATINO - MEXICAN',
            'OTHER',
            'UNABLE TO OBTAIN',
            'WHITE - RUSSIAN',
            'HISPANIC/LATINO - COLUMBIAN',
            'UNKNOWN',
            'ASIAN - SOUTH EAST ASIAN',
            'ASIAN',
            'PATIENT DECLINED TO ANSWER',
            'HISPANIC/LATINO - CUBAN',
            'WHITE - BRAZILIAN',
            'ASIAN - ASIAN INDIAN',
            'PORTUGUESE',
            'ASIAN - KOREAN',
            'HISPANIC/LATINO - DOMINICAN',
            'HISPANIC/LATINO - SALVADORAN',
            'HISPANIC/LATINO - PUERTO RICAN',
            'SOUTH AMERICAN',
            'WHITE - EASTERN EUROPEAN',
            'ASIAN - CHINESE',
            'AMERICAN INDIAN/ALASKA NATIVE',
            'HISPANIC/LATINO - CENTRAL AMERICAN',
            'MULTIPLE RACE/ETHNICITY',
            'BLACK/AFRICAN AMERICAN',
            'HISPANIC/LATINO - GUATEMALAN'
        ]
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in all_ethnicities:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def __call__(
            self,
            age: str,
            gender: str,
            ethnicity: str,
            types: List[str],
            codes: List[str]
    ):
        age = str(int(age) // 10 * 10)
        age = torch.tensor(self.age_vocabs(age))
        gender = torch.tensor(self.gender_vocabs(gender))
        ethnicity = torch.tensor(self.ethnicity_vocabs(ethnicity))
        types = to_index(types, self.type_vocabs, prefix="<cls>", suffix="<eos>")
        codes = to_index(codes, self.code_vocabs, prefix="<cls>", suffix="<eos>")
        return age, gender, ethnicity, types, codes


class eICUTokenizer:
    def __init__(self):
        self.code_vocabs, self.code_vocabs_size, self.code_embeddings = self._load_code_vocabs()
        self.type_vocabs, self.type_vocabs_size = self._load_type_vocabs()
        self.age_vocabs, self.age_vocabs_size = self._load_age_vocabs()
        self.gender_vocabs, self.gender_vocabs_size = self._load_gender_vocabs()
        self.ethnicity_vocabs, self.ethnicity_vocabs_size = self._load_ethnicity_vocabs()

    def _load_code_vocabs(self):
        vocab_dir = os.path.join(processed_data_path, f"eicu/vocab.pkl")
        vocabs = load_pickle(vocab_dir)
        vocabs_size = len(vocabs)
        embeddings = read(os.path.join(processed_data_path, f"eicu/embeddings.txt"))
        return vocabs, vocabs_size, embeddings

    def _load_type_vocabs(self):
        vocabs = Vocabulary()
        for word in [
            "diagnosis",
            "treatment",
            "medication",
        ]:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_age_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in range(18, 90):
            word = word // 10 * 10
            vocabs.add_word(str(word))
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_gender_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in ["Female", "Male", "Other", "Unknown", ""]:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_ethnicity_vocabs(self):
        # no special token needed
        vocabs = Vocabulary(init_words=[])
        for word in ["African American", "Asian", "Caucasian", "Hispanic", "Native American", "Other/Unknown", ""]:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def __call__(
            self,
            age: str,
            gender: str,
            ethnicity: str,
            types: List[str],
            codes: List[str]
    ):
        age = str(int(age) // 10 * 10)
        age = torch.tensor(self.age_vocabs(age))
        gender = torch.tensor(self.gender_vocabs(gender))
        ethnicity = torch.tensor(self.ethnicity_vocabs(ethnicity))
        types = to_index(types, self.type_vocabs, prefix="<cls>", suffix="<eos>")
        codes = to_index(codes, self.code_vocabs, prefix="<cls>", suffix="<eos>")
        return age, gender, ethnicity, types, codes
