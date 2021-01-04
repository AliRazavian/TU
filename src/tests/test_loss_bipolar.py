import unittest
import torch
import numpy as np
from Graphs.Losses.bipolar_margin_loss import Bipolar_Margin_Loss
from tests.helpers.ConfigMock import ConfigMock
from tests.helpers.ExperimentSetMock import ExperimentSetMock


class TestBipolarMetric(unittest.TestCase):

    def getMetric(self, loss_weight=torch.FloatTensor([0.5, 0.5])):
        ConfigMock({'signal_to_noise_ratio': 0.9}),
        return Bipolar_Margin_Loss(
            experiment_set=ExperimentSetMock(),
            graph=None,
            graph_cfgs={},
            loss_name='Bipolar_test',
            loss_cfgs={'loss_weight': loss_weight},
            task_cfgs={'apply': {}},
            scene_cfgs={},
            scenario_cfgs={},
            pseudo_mask_margin=0.5,
        )

    def test_perfect_match(self):
        perfect_output = torch.FloatTensor([
            [1],
            [1],
            [-1],
            [-1],
        ])
        perfect_target = torch.IntTensor([1, 1, -1, -1])
        metric = self.getMetric()
        metric.margin = 1
        loss = metric.calculate_loss(
            output=perfect_output,
            target=perfect_target,
            pos_loss_weight=0.5,
            neg_loss_weight=0.5,
        )
        self.assertEqual(loss.item(), 0)

        pseudo_loss = metric.calculate_pseudo_loss(
            output=perfect_output,
            pseudo_output=torch.randn(*perfect_output.shape),
            target=perfect_target,
        )
        self.assertEqual(pseudo_loss, 0, 'The pseudo loss is ignored when no labels are ignored')

    def test_perfect_match_multi(self):
        perfect_output = torch.FloatTensor([
            [1, 1],
            [-1, -1],
        ])
        perfect_target = torch.IntTensor([[1, 1], [-1, -1]])
        metric = self.getMetric(loss_weight=np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32))
        metric.margin = 1
        loss = metric.calculate_loss(
            output=perfect_output,
            target=perfect_target,
            pos_loss_weight=0.5,
            neg_loss_weight=0.5,
        )
        self.assertEqual(loss.item(), 0)

        pseudo_loss = metric.calculate_pseudo_loss(
            output=perfect_output,
            pseudo_output=torch.randn(*perfect_output.shape),
            target=perfect_target,
        )
        self.assertEqual(pseudo_loss, 0, 'The pseudo loss is ignored when no labels are ignored')

    def test_terrible_match(self):
        metric = self.getMetric()
        metric.margin = 0

        terrible_output = torch.FloatTensor([
            [1],
            [1],
            [-1],
            [-1],
        ])
        terrible_target = torch.IntTensor([-1, metric.ignore_index, 1, 1])
        loss = metric.calculate_loss(
            output=terrible_output,
            target=terrible_target,
            pos_loss_weight=0.5,
            neg_loss_weight=0.5,
        )

        target_loss = 1.5 / (terrible_target != metric.ignore_index).float().sum().item()
        self.assertAlmostEqual(loss.item(), target_loss, places=4)

    def test_multi_weight(self):
        terrible_output = torch.FloatTensor([
            [1, -1],
            [1, -1],
            [1, -1],
        ])
        terrible_target = torch.IntTensor([
            [-1, 1],
            [-1, 1],
            [-1, 1],
        ])
        metric = self.getMetric(loss_weight=np.array([[0.5, 0.5], [0, 0.5]], dtype=np.float32))
        metric.margin = 0
        loss = metric.calculate_loss(output=terrible_output, target=terrible_target)
        self.assertAlmostEqual(loss.item(), (3 * 2 * 0.5) / terrible_target.shape[0], 4)

        metric = self.getMetric(loss_weight=np.array([[0, 1], [1, 0]], dtype=np.float32))
        metric.margin = 0
        loss = metric.calculate_loss(output=terrible_output, target=terrible_target)
        self.assertAlmostEqual(loss.item(), 0, places=4)

        metric = self.getMetric(loss_weight=np.array([[1, 0], [0, 1]], dtype=np.float32))
        metric.margin = 0
        loss = metric.calculate_loss(output=terrible_output, target=terrible_target)
        self.assertAlmostEqual(loss.item(), (3 * 2 * 1) / terrible_target.shape[0], places=4)

        metric = self.getMetric(loss_weight=np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float32))
        metric.margin = 0
        less_terrible_output = torch.FloatTensor([
            [-1, -1],
            [1, 1],
            [1, 1],
        ])
        loss = metric.calculate_loss(output=less_terrible_output, target=terrible_target)

        expected_total_loss = (0 * .75 + 2 * .25 + 1 * .25 + 0 * .75)
        self.assertAlmostEqual(
            loss.item(),
            expected_total_loss / terrible_target.shape[0],
            places=4,
        )

        terrible_target_w_missing = terrible_target.clone()
        terrible_target_w_missing[2, 0] = metric.ignore_index
        expected_total_loss = ((0 * .75 + 1 * .25) / 2 + (1 * .25 + 0 * .75) / 3)
        loss = metric.calculate_loss(output=less_terrible_output, target=terrible_target_w_missing)
        self.assertAlmostEqual(
            loss.item(),
            expected_total_loss,
            places=4,
        )

    def test_pseudo_loss_with_multi(self):
        metric = self.getMetric()
        metric.margin = 0
        target = torch.FloatTensor([
            [1, -1],
            [metric.ignore_index, metric.ignore_index],
            [metric.ignore_index, -1],
        ])
        teacher_output = torch.FloatTensor([
            [1, -1],
            [1, -1],
            [-1, -1],
        ])
        student_output = torch.FloatTensor([
            [-1, -1],
            [1, 1],
            [1, 1],
        ])
        loss = metric.calculate_pseudo_loss(output=teacher_output, target=target, pseudo_output=student_output)

        # There are two cases in first instance but one bad == 1/2 and one out of one in the second column
        expected_total_loss = (1 / 2 * .5 + 1 / 1 * .5)
        self.assertAlmostEqual(
            loss.item(),
            expected_total_loss,
            places=4,
        )

    def test_soso_match(self):
        soso_output = torch.FloatTensor([
            [.3],
            [.1],
            [-.8],
            [-.2],
        ])
        soso_target = torch.IntTensor([1, -1, -1, 1])
        metric = self.getMetric()
        loss = metric.calculate_loss(
            output=soso_output,
            target=soso_target,
            pos_loss_weight=0.5,
            neg_loss_weight=0.5,
        )

        self.assertGreater(loss, 0)

    def test_multi_soso_match(self):
        soso_output = torch.FloatTensor([
            [.3, .1, -1],
            [-.8, -.2, 1],
        ]).T
        soso_target = torch.IntTensor([
            [1, -1, -1],
            [-1, 1, 0],
        ]).T
        metric = self.getMetric(loss_weight=np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32))
        loss = metric.calculate_loss(
            output=soso_output,
            target=soso_target,
        )

        self.assertGreater(loss, 0)

    def test_pseudo_basic(self):
        regular_output = torch.FloatTensor([
            [.3],
            [.1],
            [-.8],
            [-.2],
        ])
        pseudo_output = torch.FloatTensor([
            [.3],
            [.1],
            [-.8],
            [-.2],
        ])
        regular_target = torch.IntTensor([1, -100, -100, 1])
        metric = self.getMetric()
        loss = metric.calculate_pseudo_loss(
            output=regular_output,
            pseudo_output=pseudo_output,
            target=regular_target,
        )

        self.assertGreater(loss, 0)

    def test_pseudo_none(self):
        regular_output = torch.FloatTensor([
            [.3],
            [.1],
            [-.8],
            [-.2],
        ])
        pseudo_output = torch.FloatTensor([
            [.3],
            [.1],
            [-.8],
            [-.2],
        ])
        regular_target = torch.IntTensor([1, -1, -1, 1])
        metric = self.getMetric()
        loss = metric.calculate_pseudo_loss(
            output=regular_output,
            pseudo_output=pseudo_output,
            target=regular_target,
        )

        self.assertEqual(loss, 0)
        self.assertLess(loss, 2)

    def test_pseudo_positive(self):
        regular_output = torch.FloatTensor([
            [12312],
            [1123],
            [123],
            [1233],
        ])
        pseudo_output = torch.FloatTensor([
            [.3],
            [-1],
            [-.8],
            [-.2],
        ])
        regular_target = torch.IntTensor([-100, -100, -100, -100])
        metric = self.getMetric()
        loss = metric.calculate_pseudo_loss(
            output=regular_output,
            pseudo_output=pseudo_output,
            target=regular_target,
        )

        self.assertGreater(loss, 0)
        self.assertLess(loss, 2)

    def test_pseudo_negative(self):
        regular_output = torch.FloatTensor([
            [-12],
            [-12],
            [-12],
            [-12],
        ])
        pseudo_output = torch.FloatTensor([
            [.3],
            [.1],
            [-.8],
            [-.2],
        ])
        regular_target = torch.IntTensor([-100, -100, -100, -100])
        metric = self.getMetric()
        loss = metric.calculate_pseudo_loss(
            output=regular_output,
            pseudo_output=pseudo_output,
            target=regular_target,
        )

        self.assertGreater(loss, 0)
        self.assertLess(loss, 2)

    def test_pseudo_boundary(self):
        regular_output = torch.FloatTensor([
            [0],
            [0],
            [0],
            [0],
        ])
        pseudo_output = torch.FloatTensor([
            [.3],
            [.1],
            [-1.8],
            [-.2],
        ])
        regular_target = torch.IntTensor([-100, -100, -100, -100])
        metric = self.getMetric()
        loss = metric.calculate_pseudo_loss(
            output=regular_output,
            pseudo_output=pseudo_output,
            target=regular_target,
        )

        self.assertGreater(loss, 0)
        self.assertLess(loss, 2)


if __name__ == '__main__':
    unittest.main()
