import numpy as np
from abc import ABCMeta, abstractmethod
import time
from .base_modality import Base_Modality


class Base_Explicit(Base_Modality, metaclass=ABCMeta):
    """
    Every explicit modality should have three functions:
    1- get_item() samples from the modality
    2- get_implicit_modality_cfgs() returns the implicit modality that
        the explicit modality prefers
    3- get_default_model_cfgs() returns a modal config that maps explicit modality
        to it's implicit modality
    4- is_input_modality() and is_output_modality()
        Explicit modalities are either input modalities or output modalities
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if loading is process-heavy or disk-heavy,
        # this number will be updated in the inherited modality

        self.implicit_modality = None

    @abstractmethod
    def get_item(self, index: int, num_views=None, transforms=None):
        pass

    @abstractmethod
    def get_default_model_cfgs(self):
        pass

    @abstractmethod
    def get_implicit_modality_cfgs(self):
        pass

    def get_implicit_modality_name(self):
        return 'implicit_%s' % (self.get_name())

    def get_model_name(self):
        return '%s_path' % (self.get_name())

    def get_model_cfgs(self):
        model_cfgs = self.get_cfgs('model_cfgs')
        if (model_cfgs is None):
            model_cfgs = self.get_default_model_cfgs()
        if (self.is_input_modality()):
            model_cfgs['heads'] = [self.get_name()]
            model_cfgs['tails'] = [self.get_implicit_modality_name()]
        elif (self.is_output_modality()):
            model_cfgs['tails'] = [self.get_name()]
            model_cfgs['heads'] = [self.get_implicit_modality_name()]

        return model_cfgs

    def get_batch(self, batch):
        start_time = time.time()

        indices = batch['indices']
        num_views = batch['num_views']

        if 'spatial_transforms' not in batch:
            spatial_transforms = None
        else:
            spatial_transforms = batch['spatial_transforms']

        # indices is a one dimensional vector of numbers with batch_size elements
        # sub_indices is a two dimensional vector or numbers with batch_size x num_views_per_sample
        # transforms is a tensor of batch_size x num_views_per_sample x num_jitters x 3 x 3
        # the output is going to be batch_size x modality_size
        def partial_get_item(i):
            if spatial_transforms is None:
                return self.get_item(indices[i], num_views[i])
            else:
                return self.get_item(indices[i], num_views[i], spatial_transforms[i])

        # TODO: Simplify now that DataLoader handles the parallel data loading
        raw_data = [partial_get_item(i) for i in range(len(indices))]

        r = {k: np.concatenate([dic[k] for dic in raw_data]) for k in raw_data[0]}

        # batch.update(r)
        batch['time']['load'][self.get_name()] = {'start': start_time, 'end': time.time()}
        return r

    def is_explicit_modality(self):
        return True

    def is_csv(self):
        return False

    def is_distribution(self):
        return False
