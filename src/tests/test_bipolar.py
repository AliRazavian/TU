import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd
from collections import defaultdict
from Datasets.helpers import Dictionary_Generator
from Datasets.Modalities.bipolar import Bipolar
from tests.helpers.ConfigMock import ConfigMock


class TestBipolar(unittest.TestCase):

    def getBipolar(self, content):
        dg = Dictionary_Generator()
        dg.append_values(modality_name='test', values=content)
        ConfigMock()

        return Bipolar(
            dataset_name='test_ds',
            dataset_cfgs={},
            experiment_name='test_exp',
            experiment_cfgs={},
            modality_name='test_modality',
            content=content,
            modality_cfgs={},
            dictionary=dg.get_bipolar_dictionary('test'),
        )

    def test_label_conversion(self):
        content = pd.Series(['yes', 'yes', 'no', 'no', 'maybe', 'maybe', 'bad label should be missing'],
                            index=[[0, 0, 1, 1, 2, 2, 2], [0, 1, 0, 1, 0, 1, 2]])
        modality = self.getBipolar(content=content)
        npt.assert_array_equal(modality.labels.values, np.array([1, 1, -1, -1, 0, 0, -100]))
        npt.assert_array_equal(modality.get_loss_weight().cpu().numpy(), np.array([0.5, 0.5]))

        content = pd.Series(['yes', 'yes', 'no', 'no', 'yes', 'yes', 'yes', -100],
                            index=[[0, 0, 1, 1, 2, 2, 2, 3], [0, 1, 0, 1, 0, 1, 2, 0]])
        modality = self.getBipolar(content=content)
        npt.assert_array_almost_equal(modality.get_loss_weight().cpu().numpy(),
                                      np.array([(2 + 1) / (3 + 2), (1 + 1) / (3 + 2)]),
                                      decimal=4)

    def test_label_item_retrieval(self):
        content = pd.Series(['yes', 'yes', 'no', 'no', 'maybe', 'maybe', 'maybe'],
                            index=[[0, 0, 1, 1, 2, 2, 2], [0, 1, 0, 1, 0, 1, 2]])
        modality = self.getBipolar(content=content)
        item = modality.get_item(0)
        self.assertEqual(item['target_test_modality'], 1)
        item = modality.get_item(1)
        self.assertEqual(item['target_test_modality'], -1)
        item = modality.get_item(2)
        self.assertEqual(item['target_test_modality'], 0)

    def test_performance_computation(self):
        content = pd.Series(['yes', 'yes', 'no', 'no', 'maybe', 'maybe', 'bad label should be missing'],
                            index=[[0, 0, 1, 1, 2, 2, 2], [0, 1, 0, 1, 0, 1, 2]])
        modality = self.getBipolar(content=content)

        targets = np.array([-1, -1, 1, 1])
        perfect = modality.compute_performance(outputs=targets, targets=targets, prefix='')
        self.assertEqual(perfect['auc'], 1)
        self.assertEqual(perfect['sensitivity'], 1)
        self.assertEqual(perfect['specificity'], 1)
        self.assertEqual(perfect['precision'], 1)
        self.assertEqual(perfect['negative_predictive_value'], 1)

        terrible = modality.compute_performance(outputs=np.array([1, 1, -1, -1]), targets=targets, prefix='')
        self.assertEqual(terrible['auc'], 0)
        self.assertEqual(terrible['sensitivity'], 0)
        self.assertEqual(terrible['specificity'], 0)
        self.assertEqual(terrible['precision'], 0)
        self.assertEqual(terrible['negative_predictive_value'], 0)

        only_false = modality.compute_performance(outputs=np.array([-1, -1, -1, -1]),
                                                  targets=np.array([-1, -1, -1, -1]),
                                                  prefix='')
        npt.assert_equal(only_false['auc'], np.nan)
        npt.assert_equal(only_false['sensitivity'], np.nan)
        self.assertEqual(only_false['specificity'], 1)
        npt.assert_equal(only_false['precision'], np.nan)
        self.assertEqual(only_false['negative_predictive_value'], 1)

        only_true = modality.compute_performance(outputs=np.array([1, 1, 1, 1]),
                                                 targets=np.array([1, 1, 1, 1]),
                                                 prefix='')
        npt.assert_equal(only_true['auc'], np.nan)
        self.assertEqual(only_true['sensitivity'], 1)
        npt.assert_equal(only_true['specificity'], np.nan)
        self.assertEqual(only_true['precision'], 1)
        npt.assert_equal(only_true['negative_predictive_value'], np.nan)

        mixed = modality.compute_performance(outputs=np.array([-.1, -.2, 1, 1, 1]),
                                             targets=np.array([1, -1, -1, 1, 0]),
                                             prefix='prefix_test_')
        self.assertGreater(mixed['prefix_test_auc'], 0.5)
        self.assertLess(mixed['prefix_test_sensitivity'], 1)
        self.assertLess(mixed['prefix_test_specificity'], 1)
        self.assertLess(mixed['prefix_test_precision'], 1)
        self.assertLess(mixed['prefix_test_negative_predictive_value'], 1)

    def test_report_computation_with_missing(self):
        content = pd.Series(['yes', 'yes', 'no', 'no', 'maybe', 'maybe', 'bad label should be missing'],
                            index=[[0, 0, 1, 1, 2, 2, 2], [0, 1, 0, 1, 0, 1, 2]])
        modality = self.getBipolar(content=content)
        targets = [modality.get_item(i)[modality.get_batch_name()] for i in range(3)]
        targets = np.concatenate(targets)

        outputs = targets.copy()
        batch = {'indices': [0, 1, 2], 'results': defaultdict(dict)}
        batch['results']['test_modality'].update({
            'output': outputs,
            'target': targets,
        })
        modality.analyze_modality_specific_results(batch)
        summary = {'modalities': defaultdict(dict)}
        modality.report_modality_specific_epoch_summary(summary)

        perfect = summary['modalities']['test_modality']
        self.assertEqual(perfect['accuracy'], 1)
        self.assertEqual(perfect['specificity'], 1)
        self.assertEqual(perfect['negative_predictive_value'], 1)

        outputs[1] = outputs[1] * -1
        batch['results']['test_modality'].update({
            'output': outputs,
            'target': targets,
        })
        modality.analyze_modality_specific_results(batch)
        summary = {'modalities': defaultdict(dict)}
        modality.report_modality_specific_epoch_summary(summary)

        almost_perfect = summary['modalities']['test_modality']
        self.assertEqual(almost_perfect['accuracy'], 0.5)


if __name__ == '__main__':
    unittest.main()
