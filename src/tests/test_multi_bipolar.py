import unittest
from collections import defaultdict
import numpy as np
import numpy.testing as npt
import pandas as pd
from Datasets.helpers import Dictionary_Generator
from Datasets.Modalities.multi_bipolar import Multi_Bipolar
from tests.helpers.ConfigMock import ConfigMock


class TestMultiBipolar(unittest.TestCase):

    def getMultiBipolar(self, content):
        dg = Dictionary_Generator()
        for column_name in content:
            dg.append_values(modality_name='test', values=content[column_name])

        ConfigMock()
        return Multi_Bipolar(
            dataset_name='test_ds',
            dataset_cfgs={},
            experiment_name='test_exp',
            experiment_cfgs={},
            modality_name='test_modality',
            content=content,
            modality_cfgs={
                'columns': content.columns.to_list(),
                'to_each_view_its_own_label': False,
                'skip_dictionary_save': True,
            },
            dictionary=dg.get_bipolar_dictionary('test'),
        )

    def test_label_conversion(self):
        content = pd.DataFrame(
            {
                'col1': ['yes', 'yes', 'no', 'no', 'maybe', 'maybe', 'bad label should be missing'],
                'col2': ['yes', 'yes', 'yes', 'yes', 'no', 'no', 'no'],
            },
            index=[[0, 0, 1, 1, 2, 2, 2], [0, 1, 0, 1, 0, 1, 2]])

        modality = self.getMultiBipolar(content=content)
        npt.assert_array_equal(
            modality.labels.values,
            np.array([[1, 1, -1, -1, 0, 0, -100], [1, 1, 1, 1, -1, -1, -1]]).T,
        )
        weights = modality.get_loss_weight()
        npt.assert_array_almost_equal(
            weights.cpu().numpy(),
            np.array([[0.5, 0.5], [(2 + 1) / (3 + 2), (1 + 1) / (3 + 2)]]),
        )

    def test_label_item_retrieval(self):
        content = pd.DataFrame(
            {
                'col1': ['yes', 'yes', 'no', 'no', 'maybe', 'maybe', 'maybe'],
                'col2': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
            },
            index=[[0, 0, 1, 1, 2, 2, 2], [0, 1, 0, 1, 0, 1, 2]])
        modality = self.getMultiBipolar(content=content)
        item = modality.get_item(0)
        npt.assert_array_equal(item['target_test_modality'], np.array([[1, -1]]))
        item = modality.get_item(1)
        npt.assert_array_equal(item['target_test_modality'], np.array([[-1, 1]]))
        item = modality.get_item(2)
        npt.assert_array_equal(item['target_test_modality'], np.array([[0, 1]]))

    def test_report_computation(self):
        content = pd.DataFrame(
            {
                'col1': ['yes', 'yes', 'no', 'no', 'maybe', 'maybe', 'maybe'],
                'col2': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
            },
            index=[[0, 0, 1, 1, 2, 2, 2], [0, 1, 0, 1, 0, 1, 2]])
        modality = self.getMultiBipolar(content=content)

        targets = np.array([[1, -1], [-1, 1], [1, 1]])
        batch = {'indices': [0, 1, 2], 'results': defaultdict(dict)}
        batch['results']['test_modality'] = {'output': targets, 'target': targets}

        modality.analyze_modality_specific_results(batch)
        summary = {'modalities': defaultdict(dict)}
        modality.report_modality_specific_epoch_summary(summary)
        for perfect in summary['modalities'].values():
            self.assertEqual(perfect['auc'], 1)
            self.assertEqual(perfect['sensitivity'], 1)
            self.assertEqual(perfect['specificity'], 1)
            self.assertEqual(perfect['precision'], 1)
            self.assertEqual(perfect['negative_predictive_value'], 1)

        batch['results']['test_modality'].update({
            'output': np.array([[-1, 1], [1, -1], [-1, -1]]),
            'target': targets,
        })
        modality.analyze_modality_specific_results(batch)
        modality.report_modality_specific_epoch_summary(summary)

        for terrible in summary['modalities'].values():
            self.assertEqual(terrible['auc'], 0)
            self.assertEqual(terrible['sensitivity'], 0)
            self.assertEqual(terrible['specificity'], 0)
            self.assertEqual(terrible['precision'], 0)
            self.assertEqual(terrible['negative_predictive_value'], 0)

    def test_report_computation_with_missing(self):
        content = pd.DataFrame(
            {
                'col1': [-100, -100, 'no', 'no', 'no', 'no', 'no'],
                'col2': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
            },
            index=[[0, 0, 1, 1, 2, 2, 2], [0, 1, 0, 1, 0, 1, 2]])
        modality = self.getMultiBipolar(content=content)
        targets_missing = [modality.get_item(i)[modality.get_batch_name()] for i in range(3)]
        targets_missing = np.concatenate(targets_missing)

        outputs = targets_missing.copy()
        outputs[0, 0] = -1
        batch = {'indices': [0, 1, 2], 'results': defaultdict(dict)}
        batch['results']['test_modality'].update({
            'output': outputs,
            'target': targets_missing,
        })
        modality.analyze_modality_specific_results(batch)
        summary = {'modalities': defaultdict(dict)}
        modality.report_modality_specific_epoch_summary(summary)

        for perfect in summary['modalities'].values():
            self.assertEqual(perfect['accuracy'], 1)
            self.assertEqual(perfect['specificity'], 1)
            self.assertEqual(perfect['negative_predictive_value'], 1)

        outputs[1, 0] = outputs[1, 0] * -1
        outputs[1, 1] = outputs[1, 1] * -1
        batch['results']['test_modality'].update({
            'output': outputs,
            'target': targets_missing,
        })
        modality.analyze_modality_specific_results(batch)
        modality.report_modality_specific_epoch_summary(summary)

        almost_perfect = summary['modalities']['test_modality_col1']
        self.assertEqual(almost_perfect['accuracy'], 0.5)

        almost_perfect = summary['modalities']['test_modality_col2']
        self.assertAlmostEqual(almost_perfect['accuracy'], 2 / 3, places=4)


if __name__ == '__main__':
    unittest.main()
