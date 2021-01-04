import numbers
import numpy as np
import torch

from Graphs.Losses.helpers import normalized_euclidean_distance
from .Base_Modalities.base_number import Base_Number
from .Base_Modalities.base_output import Base_Output
from .Base_Modalities.base_implicit import Base_Implicit


# Helpers for dim bug
def check_dims(var):
    if (not isinstance(var, torch.Tensor) and not isinstance(var, np.ndarray)):
        return False

    if var.shape == ():
        return False

    return True


def fix_dims(var):
    if check_dims(var):
        return var
    return np.array([var])


class ID_from_Indices(Base_Number, Base_Output, Base_Implicit):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels = self.get_cfgs('num_channels', default=128)
        # Content is a pd.Series with -100 defined in get_modality_and_content
        self.content = kwargs.pop('content')

    def has_classification_loss(self):
        return False

    def has_identification_loss(self):
        return True

    def has_reconstruction_loss(self):
        return False

    def get_identification_loss_name(self):
        return '%s_triplet_metric' % self.get_name()

    def get_identification_loss_cfgs(self):
        return {
            'loss_type': 'triplet_metric',
            'modality_name': self.get_name(),
            'output_name': self.get_encoder_name(),
            'target_name': self.get_batch_name(),
            'output_shape': self.get_shape()
        }

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'fully_connected',
                'num_hidden': 0,
            }
        }

    def get_implicit_modality_cfgs(self):
        return {
            'type': 'implicit',
            'num_channels': self.num_channels,
            'explicit_modality': self.get_name(),
            'has_reconstruction_loss': False,
        }

    def get_item(self, index: int, num_views: int, transforms=None):
        return {self.get_batch_name(): np.array([index] * (num_views * self.num_jitters))}

    def analyze_modality_specific_results(self, batch):
        distance = self.unwrap(batch['results'][self.get_name()].pop('euclidean_distance'))
        target = self.unwrap(batch['results'][self.get_name()].pop('target'))

        accuracy = self.compute_accuracy(distance=distance, target=target)
        self.set_runtime_value(
            runtime_value_name='accuracy',
            value=accuracy,
            indices=batch['indices'],
            sub_indices=batch['sub_indices'],
        )

        batch['results'][self.get_name()].update({'accuracy': accuracy.mean()})
        pass

    def compute_accuracy(self, output=None, target=None, distance=None):
        """
        Compute the accuracy for the euclidean distance

        See if the closes n-1 images are correct

        The inputs are numpy tensors of dimension:
        - output.shape = [batch_size, 256]
        - target.shape = [batch_size, 1]
        """
        assert output is not None or distance is not None, 'Either output or distance is required'
        assert target is not None, 'Target is required despite the None in the parameters'
        if distance is None:
            distance = self.unwrap(normalized_euclidean_distance(self.wrap(output)))

        target = target.repeat(axis=1, repeats=target.shape[0])
        target = target == target.T
        tsum = target.sum(axis=1)

        tsort_idx = target.argsort(axis=1)[:, ::-1]  # Sorts descending
        dist_idx = distance.argsort(axis=1)

        matches = [(tsort_idx[i, :tsum[i]] == np.sort(dist_idx[i, :tsum[i]])[::-1]) for i in range(tsum.shape[0])]
        # We need to subtract self as this will always match and is trivial
        accuracy = [max(0, match.sum() - 1) / (match.shape[0] - 1.0) for match in matches if match.shape[0] > 1]
        if len(accuracy) == 0:
            return None

        return np.stack(accuracy)

    def get_runtime_value(self, runtime_value_name):
        """
        runtime values are the values that are computed during
        the runtime, like accuracy, etc...
        We store them during training and testing to be able to
        measure performance.
        """
        name = runtime_value_name.lower()

        if name not in self.runtime_values:
            self.runtime_values[name] = self.get_initial_runtime_value(runtime_value_name=runtime_value_name)

        return self.runtime_values[name]

    def get_default_value(self, runtime_value_name: str):
        name = runtime_value_name.lower()
        if name.endswith('accuracy'):
            return self.get_cfgs('ignore_index', default=-100.)
        else:
            raise BaseException('Unknown runtime %s' % (runtime_value_name))

    def get_initial_runtime_value(self, runtime_value_name: str):
        runtime_value_name = runtime_value_name.lower()
        assert (runtime_value_name not in self.runtime_values),\
            ''.join(['Trying to init %s but it is ',
                     'already initialized in %s']) % (
            runtime_value_name, self.get_name())

        runtime_value = self.content.copy()
        runtime_value.name = '%s_%s' % (self.get_name(), runtime_value_name)

        default_value = self.get_default_value(runtime_value_name=runtime_value_name)
        if isinstance(default_value, (numbers.Number, bool, str)):
            runtime_value.values.fill(default_value)
        elif isinstance(default_value, dict):
            runtime_value = runtime_value.map(default_value)
        return runtime_value

    def set_runtime_value(self, runtime_value_name, value, indices, sub_indices):
        runtime_value = self.get_runtime_value(runtime_value_name)
        value = fix_dims(value)
        val_idx = 0
        for i in range(len(indices)):
            if (len(sub_indices[i]) == 1):
                # There is no stats for 0 len indices
                continue

            for sub_idx in sub_indices[i]:
                if val_idx > len(value):
                    raise IndexError(f'Tried to index {val_idx} from {len(value)} array')

                runtime_value[indices[i]][sub_idx] = value[val_idx]
                val_idx += 1
