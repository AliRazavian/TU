from collections import defaultdict
import pandas as pd
import numpy as np
from GeneralHelpers import Singleton
from file_manager import File_Manager
from UIs.console_UI import Console_UI


def build_empty_dictionary(values: list):
    empty_url = 'https://babelnet.org/synset?word='
    empty_desc = 'Unknown description for '

    return pd.DataFrame({
        'label': range(len(values)),
        'name': values,
        'url': [f'{empty_url}{val}' for val in values],
        'desc': [f'{empty_desc}{val}' for val in values],
    })


class Dictionary_Generator(metaclass=Singleton):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dictionary = {}
        self.__values = defaultdict(list)
        self.__suggested_dictionaries = defaultdict(list)

    def get(self, modality_name: str, action_on_missing='exception'):
        modality_name = modality_name.lower()
        if modality_name in self.__dictionary:
            return self.__dictionary[modality_name]

        suggested = self.get_merged_suggested_dictionary(modality_name=modality_name)
        if suggested is not None:
            self.check_suggested_dictionary(modality_name=modality_name,
                                            dictionary=suggested,
                                            action_on_missing=action_on_missing)
            return suggested

        if modality_name not in self.__values:
            return None

        self.__dictionary[modality_name] = build_empty_dictionary(values=self.__values[modality_name])

        return self.__dictionary[modality_name]

    def get_bipolar_dictionary(self, modality_name: str, label_dictionary: dict = None):
        if modality_name not in self.__values:
            return None

        if label_dictionary is not None:
            assert isinstance(label_dictionary, dict)
            for t in ['positive', 'negative', 'boundary']:
                if t not in label_dictionary:
                    if t != 'boundary':
                        raise IndexError(f'The dictionary doesn\'t have a {t} index: {label_dictionary}')
                else:
                    if isinstance(label_dictionary[t], str):
                        label_dictionary[t] = [label_dictionary[t]]

                    assert isinstance(label_dictionary[t], list), f'Expected {t} to be a list, got {label_dictionary}'
        else:
            label_dictionary = {
                'positive': ['yes', 'y', 'yup', 'ja', 'positive', 'pos', 'positiv', 'jakande', '1', 'true', True],
                'negative': ['no', 'n', 'nope', 'nej', 'negative', 'neg', 'negativ', 'nekande', '-1', 'false', False],
                'boundary': ['maybe', 'perhaps', 'possibly', 'hard to tell', 'hard_to_tell', 'kanske', '0'],
            }

        all_labels = {}
        for pos in label_dictionary['positive']:
            all_labels[pos] = 1
        for neg in label_dictionary['negative']:
            all_labels[neg] = -1
        if 'boundary' in label_dictionary:
            for bdr in label_dictionary['boundary']:
                all_labels[bdr] = 0

        unique_classes = self.__values[modality_name]
        if len(unique_classes) <= 1:
            raise ValueError(f'The modality {modality_name} has {len(unique_classes)} unique classes (min 2)')

        label_to_cls_name = {}
        for u in unique_classes:
            if (u in all_labels):
                label_to_cls_name[all_labels[u]] = u

        dictionary = {
            'label': [k for k, v in label_to_cls_name.items()],
            'name': [v for k, v in label_to_cls_name.items()],
            'url': None,
            'desc': None,
        }

        return pd.DataFrame(dictionary)

    def append_values(self, modality_name: str, values: pd.Series, ignore_index=-100):
        assert isinstance(modality_name, str), f'Modality name is not a string "{modality_name}"'
        assert isinstance(values, pd.Series), 'All values should be a pandas Series object'
        modality_name = modality_name.lower()

        values2ignore = [
            # "Regular" missing values
            'nan',
            '__nan__',
            np.nan,
            '',
            '_',
            'unknown',
            float('nan'),
            None,
            'none',
            'na',
            ignore_index,
            str(ignore_index),
            "['']",
            # New values that we should skip
            # TODO: probably move this to check function or a yaml file
            'frontalvridningutt',
            'hl'
        ]

        clean_values = values.dropna()
        # Unfortunately pandas stores strings as objects so a simple dtype == str doesn't work
        # and relying on dtype == object seem like opening pandoras box...
        if clean_values.apply(type).eq(str).any():
            clean_values = values.str.lower()
        clean_values = clean_values.unique()
        [
            self.__values[modality_name].append(value)
            for value in clean_values
            if value not in self.__values[modality_name] and value not in values2ignore
        ]
        # not needed as in works for nan:
        # (type(value) != float and type(value) != int) or not np.isnan(value))

        return self

    def append_suggested_dictionary(self, dataset_name, modality_name, FMSingleton=None):
        assert isinstance(modality_name, str), f'Modality name is not a string "{modality_name}"'
        assert isinstance(dataset_name, str), f'Dataset name is not a string "{modality_name}"'
        modality_name = modality_name.lower()

        if FMSingleton is None:
            FMSingleton = File_Manager()
        suggested_dictionary = FMSingleton.read_dictionary(
            dataset_name=dataset_name,
            modality_name=modality_name,
        )
        if suggested_dictionary is not None:
            self.__suggested_dictionaries[modality_name].append(suggested_dictionary)

        return self

    def get_merged_suggested_dictionary(self, modality_name):
        modality_name = modality_name.lower()

        if modality_name not in self.__suggested_dictionaries:
            return None

        dicts = self.__suggested_dictionaries[modality_name]
        biggest_dict = None
        biggest_idx = -1
        for idx in range(len(dicts)):
            d = dicts[idx]
            if biggest_dict is None or len(d) > len(biggest_dict):
                biggest_dict = d
                biggest_idx = idx

        if biggest_idx < 0:
            raise ValueError('Could not find a proper dataframe')

        for idx in range(len(dicts)):
            if idx == biggest_idx:
                continue

            d = dicts[idx]
            if 'name' not in d:
                raise IndexError(f'Could not find name in {modality_name} among the indexes for {d.keys()}')
            mismatch = ~d['name'].isin(biggest_dict['name'])
            if any(mismatch):
                d = d[mismatch].copy()
                d['label'] = np.linspace(1, len(d), len(d), dtype=int) + max(biggest_dict['label'])
                biggest_dict = biggest_dict.append(d)

        return biggest_dict

    def check_suggested_dictionary(self, modality_name: str, dictionary: pd.DataFrame, action_on_missing: str):
        modality_name = modality_name.lower()

        if modality_name not in self.__values:
            return True

        if 'name' not in dictionary:
            # Multi-bipolar dictionaries should not be checked in this manner
            return False

        values = pd.Series(self.__values[modality_name])
        not_in_dictionary = ~values.isin(dictionary['name'])
        if any(not_in_dictionary):
            msg = f'Missing values "{values[not_in_dictionary].tolist()}" from the suggested dictionary' +\
                f' for {modality_name}'
            if action_on_missing == 'exception':
                raise IndexError(msg)
            elif action_on_missing != 'silent':
                Console_UI().inform_user(msg)

            return False

        return True
