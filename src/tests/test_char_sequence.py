import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd
from Datasets.Modalities.char_sequence import Char_Sequence
from tests.helpers.ConfigMock import ConfigMock


class TestChar_Sequence(unittest.TestCase):

    def getCharSequence(self, content, dictionary=' .,\\-abcdefghijklmnopqrstuvwxyzäåö$()'):
        ConfigMock()
        return Char_Sequence(
            dataset_name='test_ds',
            dataset_cfgs={},
            experiment_name='test_exp',
            experiment_cfgs={},
            modality_name='test_modality',
            content=content,
            modality_cfgs={'dictionary': dictionary},
        )

    def test_dictionary(self):
        content = pd.Series(
            [
                'abc',
                'åäö',
                np.nan,
                'evil var test !"#¤%%&/',
                'evil var test !"#¤%%&/',
                'evil var test !"#¤%%&/',
            ],
            index=[[0, 0, 1, 2, 2, 2], [0, 1, 0, 0, 1, 2]],
        )
        modality = self.getCharSequence(content=content, dictionary=' ab')
        self.assertEqual(max(modality.char_to_ix.values()), 2)
        self.assertEqual(len(modality.char_to_ix.values()), 3)
        self.assertEqual(modality.char_to_ix['a'], 1)

    def test_char_item_retrieval(self):
        content = pd.Series(
            [
                'abc',
                'åäö',  # Should be ignored
                np.nan,
                'evil var test !"#¤%%&/',
                'evil var test !"#¤%%&/',
                'evil var test !"#¤%%&/',
            ],
            index=[[0, 0, 1, 2, 2, 2], [0, 1, 0, 0, 1, 2]],
        )
        modality = self.getCharSequence(content=content, dictionary=' ab')

        item = modality.get_item(0)
        npt.assert_array_equal(item['encoder_test_modality'][0].shape, [1, 3, modality.sequence_length])
        self.assertEqual(
            np.sum(item['encoder_test_modality'][0][0][0]),
            modality.sequence_length - 2,
            'The length of spaces',
        )
        self.assertEqual(
            np.sum(item['encoder_test_modality'][0][0][2]),
            1,
            'The length of b',
        )
        self.assertEqual(
            np.sum(item['encoder_test_modality'][0][0][1]),
            1,
            'The length of a',
        )
        item = modality.get_item(1)
        self.assertEqual(
            np.sum(item['encoder_test_modality'][0][0][0]),
            modality.sequence_length,
            'Empty string',
        )


if __name__ == '__main__':
    unittest.main()
