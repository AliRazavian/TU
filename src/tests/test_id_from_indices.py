import unittest
import torch
import pandas as pd
from Graphs.Losses.helpers import normalized_euclidean_distance
from Datasets.Modalities.id_from_indices import ID_from_Indices
from tests.helpers.ConfigMock import ConfigMock


class TestIdFromIndices(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ConfigMock()
        self.modality = ID_from_Indices(
            dataset_name='test_ds',
            dataset_cfgs={},
            experiment_name='test_exp',
            experiment_cfgs={},
            modality_name='test_modality',
            content=pd.Series(-100, index=[[12, 12, 55, 55, 56, 56, 56], [0, 1, 0, 1, 0, 1, 2]], dtype='int64'),
            modality_cfgs={},
        )

    def get_batch(self, output, target, distance=None, indices=[12, 55], sub_indices=[[0, 1], [0, 1]]):
        if distance is None:
            distance = normalized_euclidean_distance(output)
        batch = {
            'indices': indices,
            'sub_indices': sub_indices,
            'results': {
                'test_modality': {
                    'output': output,
                    'target': target,
                    'euclidean_distance': distance,
                }
            }
        }
        return batch

    def test_perfect_match(self):
        perfect_output = torch.FloatTensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ])
        perfect_target = torch.IntTensor([12, 12, 55, 55]).view(-1, 1)
        batch = self.get_batch(
            output=perfect_output,
            target=perfect_target,
        )
        self.modality.analyze_modality_specific_results(batch)
        self.assertEqual(batch['results']['test_modality']['accuracy'], 1)

    def test_terrible_match(self):
        terrible_output = torch.FloatTensor([
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ])
        terrible_target = torch.IntTensor([12, 12, 55, 55]).view(-1, 1)
        batch = self.get_batch(
            output=terrible_output,
            target=terrible_target,
        )
        self.modality.analyze_modality_specific_results(batch)
        self.assertEqual(batch['results']['test_modality']['accuracy'], 0)

    def test_soso_match(self):
        terrible_output = torch.FloatTensor([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 2],
        ])
        terrible_target = torch.IntTensor([12, 12, 55, 55]).view(-1, 1)
        batch = self.get_batch(
            output=terrible_output,
            target=terrible_target,
        )
        self.modality.analyze_modality_specific_results(batch)
        accuracy = batch['results']['test_modality']['accuracy']
        self.assertLess(accuracy, 1)
        self.assertGreater(accuracy, 0)


if __name__ == '__main__':
    unittest.main()
