import unittest
import torch
from Graphs.Losses.triplet_metric_loss import Triplet_Metric_Loss
from tests.helpers.ConfigMock import ConfigMock
from tests.helpers.ExperimentSetMock import ExperimentSetMock


class TestTripleMetric(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ConfigMock()
        self.metric = Triplet_Metric_Loss(
            experiment_set=ExperimentSetMock(),
            graph=None,
            graph_cfgs={},
            loss_name='Test',
            loss_cfgs={},
            task_cfgs={'apply': {}},
            scene_cfgs={},
            scenario_cfgs={},
        )

    def test_perfect_match(self):
        perfect_output = torch.FloatTensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ])
        perfect_target = torch.IntTensor([12, 12, 55, 55]).view(-1, 1)
        loss = self.metric.calculate_loss(
            output=perfect_output,
            target=perfect_target,
        )
        self.assertEqual(loss, 0)

    def test_terrible_match(self):
        terrible_output = torch.FloatTensor([
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ])
        terrible_target = torch.IntTensor([12, 12, 55, 55]).view(-1, 1)
        loss = self.metric.calculate_loss(
            output=terrible_output,
            target=terrible_target,
        )
        self.assertAlmostEqual(loss, 1 + 2**0.5)

    def test_soso_match(self):
        soso_output = torch.FloatTensor([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 2],
        ])
        soso_target = torch.IntTensor([12, 12, 55, 55]).view(-1, 1)
        loss = self.metric.calculate_loss(
            output=soso_output,
            target=soso_target,
        )
        self.assertLess(loss, 1 + 2**0.5)
        self.assertGreater(loss, 0)

    def test_single_batch_type(self):
        soso_output = torch.FloatTensor([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 2],
        ])
        soso_target = torch.IntTensor([12, 12, 12, 12]).view(-1, 1)
        loss = self.metric.calculate_loss(
            output=soso_output,
            target=soso_target,
        )
        self.assertEqual(loss, 0)


if __name__ == '__main__':
    unittest.main()
