import math
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from .base_number import Base_Number
from .base_csv import Base_CSV


class Base_Label(Base_Number, Base_CSV, metaclass=ABCMeta):

    @abstractmethod
    def get_loss_type(self):
        pass

    @abstractmethod
    def collect_statistics(self):
        pass

    def init_dictionary(self):
        pass

    @abstractmethod
    def get_num_classes(self):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = self.get_cfgs('ignore_index', default=-100)
        self.signal_to_noise_ratio = self.get_cfgs('signal_to_noise_ratio', default=1)
        self.jitter_pool = self.get_cfgs('jitter_pool', default='mean')
        self.view_pool = self.get_cfgs('view_pool', default='max')
        self.to_each_view_its_own_label = self.get_cfgs('to_each_view_its_own_label', default=False)

        self.criterion_weight = None
        self.dictionary = None
        self.proprity_views = []

        self.label_to_url = {}
        self.label_to_desc = {}

        self.label_stats = {}

        self.dictionary = kwargs['dictionary']
        self.init_dictionary()

        self.prep_content()
        self.convert_class_names_to_indices()
        self.collect_statistics()

        no = self.get_num_classes()
        self.set_channels(no)

        self.init_priority_views()

    def has_classification_loss(self):
        return True

    def get_initial_informativeness(self):
        if self.dictionary is None:
            raise Exception(f'Dictionary has not been generated as expected for {self.get_name()}')

        if 'label_informativeness' not in self.label_stats:
            self.collect_statistics()

        label_informativeness = {}
        for label_name, label_index in self.cls_name_to_label.items():
            label_informativeness[label_name] = self.label_stats['label_informativeness'][label_index]

        return label_informativeness

    def prep_content(self):
        self.content = self.content.apply(lambda x: x.lower() if (isinstance(x, str)) else x)

    def convert_class_names_to_indices(self):
        if self.dictionary is None:
            raise Exception(f'No dictionary has been generated for {self.get_name()}')
        if len(self.cls_name_to_label) < 2:
            raise Exception(f'No valid conversion has been generated for {self.get_name()}: {self.cls_name_to_label}')

        self.labels = self.content.map(self.cls_name_to_label).fillna(self.ignore_index).astype(int)
        assert self.labels.dtype in ['int'], f'Unknown label type "{str(self.labels.dtype)}"'

        return self

    @property
    def cls_name_to_label(self):
        assert isinstance(self.dictionary, pd.DataFrame), 'No dictionary set while trying to retrieve class name'

        cls_name_to_label = {}
        for i in range(len(self.dictionary)):
            name = self.dictionary['name'].iloc[i]
            label = self.dictionary['label'].iloc[i]

            cls_name_to_label[str(name).lower()] = label

        return cls_name_to_label

    def has_pseudo_label(self):
        return True

    def get_label_dictionary(self):
        if self.dictionary is None:
            self.init_dictionary()
        return self.cls_name_to_label

    def get_item(self, index, num_view=None, spatial_transfrom=None):
        if self.to_each_view_its_own_label:
            labels = np.array(self.labels[index])
        else:
            # All labels are assumed to be identical, hence [0]
            labels = np.array(self.labels[index][0]).reshape(-1,)
        # TODO: check that labels is float32 | int

        return {self.get_batch_name(): labels}

    def init_priority_views(self):
        priority_classes = self.get_cfgs('priority_sampling_from_classes', default=[])
        self.priority_labels = [self.cls_name_to_label[str(s).lower()] for s in priority_classes]

    def get_classification_loss_cfgs(self):
        return {
            'loss_type': self.get_loss_type(),
            'modality_name': self.get_name(),
            'output_name': self.get_encoder_name(),
            'target_name': self.get_batch_name(),
            'loss_weight': self.get_loss_weight(),
            'ignore_index': self.ignore_index,
            'jitter_pool': self.jitter_pool,
            'view_pool': self.view_pool,
            'signal_to_noise_ratio': self.signal_to_noise_ratio,
            'to_each_view_its_own_label': self.to_each_view_its_own_label,
            'output_shape': self.get_shape()
        }

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'fully_connected',
                'num_hidden': 1,
            }
        }

    def get_implicit_modality_cfgs(self):
        nc = max(
            self.get_cfgs('min_channel', default=2),
            2**math.ceil(math.log2(self.get_num_classes())),
        )
        return {
            'type': 'implicit',
            'num_channels': nc,
            'explicit_modality': self.get_name(),
            'has_reconstruction_loss': False,
        }

    def get_loss_weight(self):
        return self.wrap(self.label_stats['loss_weight'])

    def get_mean_accuracy(self, unfiltered_accuracy):
        accuracy = unfiltered_accuracy[unfiltered_accuracy != self.ignore_index]
        if len(accuracy) == 0:
            return None
        return np.mean(accuracy)

    def is_classification(self):
        return True

    def is_regression(self):
        return False
