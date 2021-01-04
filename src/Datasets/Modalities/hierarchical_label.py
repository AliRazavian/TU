import numpy as np
import pandas as pd
import itertools

from file_manager import File_Manager
from .Base_Modalities.base_label import Base_Label
from .Base_Modalities.base_output import Base_Output


class Hierarchical_Label(Base_Label, Base_Output):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_indices = None
        self.__cache = {}

    def make_dictionary(self):
        unique = self.content.str.lower().dropna().unique()
        self.trie = self.cast_unique_contents_to_trie(sorted(unique, key=len))
        self.hierarchical_content = self.get_hierarchical_content(self.trie)
        return pd.DataFrame(self.hierarchical_content)

    def convert_class_names_to_indices(self):
        self.max_levels = self.dictionary['level'].max()
        self.num_levels = self.max_levels + 1
        unique_names = sorted(self.dictionary['name'].unique(), key=len)
        num_classes = self.get_num_classes()
        self.cls_name_to_label = {'': np.zeros(num_classes, dtype='float32')}
        for u in unique_names:
            self.cls_name_to_label[u] = np.zeros(num_classes, dtype='float32')
            self.cls_name_to_label[u][unique_names.index(u)] = 1
            self.cls_name_to_label[u] += self.cls_name_to_label[u[:-1]]

        self.cls_name_to_label[np.nan] = np.zeros(num_classes, dtype='float32')
        self.labels = self.content.map(self.cls_name_to_label)

    def collect_statistics(self):
        self.label_stats = {}
        # v = self.labels.values

    def get_num_classes(self):
        if self.dictionary is None:
            raise Exception(f'No dictionary has been initated for {self.get_name()}')

        if 'num_classes' not in self.__cache:
            unique_names = sorted(self.dictionary['name'].unique(), key=len)
            self.__cache['num_classes'] = len(unique_names)

        return self.__cache['num_classes']

    def get_loss_type(self):
        return 'hierarchical_bce'

    def set_runtime_value(self, runtime_value_name, value, indices, sub_indices):
        pass

    def cast_unique_contents_to_trie(self, unique_contents, trie={}):
        if isinstance(unique_contents, list):
            for u in unique_contents:
                self.cast_unique_contents_to_trie(u, trie)
        else:
            u = unique_contents
            if (len(u) == 0):
                return
            if (u[:-1] in trie):
                trie[u[:-1]].append(u)
            else:
                trie[u[:-1]] = [u]
            self.cast_unique_contents_to_trie(u[:-1], trie)
            trie[u[:-1]] = sorted(list(set(trie[u[:-1]])))
        return trie

    def get_hierarchical_content(self, trie, roots=[''], ret=[], level=0):
        current_level = [trie[r] if r in trie else [r] for r in roots]
        current_level = list(itertools.chain(*current_level))
        if (current_level == roots):
            return ret

        current_width = [len(trie[r]) if r in trie else 1 for r in current_level]
        max_pools = np.cumsum(current_width)

        for i, k, p in zip(np.arange(len(current_level)), current_level, max_pools):
            ret.append({'name': k, 'level': level, 'slice_indices': p, 'label': i})

        return self.get_hierarchical_content(trie, current_level, ret, level + 1)

    def init_dictionary(self):
        if self.dictionary is None:
            fm = File_Manager()
            self.dictionary = fm.read_dictionary(dataset_name=self.dataset_name, modality_name=self.get_name())
            if self.dictionary is None:
                self.dictionary = self.make_dictionary()  # no dictionary
                fm.write_dictionary(dictionary=self.dictionary,
                                    dataset_name=self.dataset_name,
                                    modality_name=self.get_name())
            else:
                fm.write_dictionary2logdir(dictionary=self.dictionary, modality_name=self.get_name())

    def get_loss_weight(self):
        return None

    # def analyze_modality_specific_results(self, batch):
    #     results = {}  # noqa: F841
    #     output = batch['results'][self.get_name()].pop('output')
    #     output = self.unwrap(output)
    #     output = output.squeeze()
    #     target = self.unwrap(batch['results'][self.get_name()].pop('target')).reshape(-1)  # noqa: F841
    #     if ('pseudo_output' in batch['results'][self.get_name()]):
    #         pseudo_output = self.unwrap(batch['results'][self.get_name()].pop('pseudo_output'))  # noqa: F841
