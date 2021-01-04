import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd
from Datasets.helpers import Dictionary_Generator, build_empty_dictionary
from tests.helpers.FileManagerMock import FileManagerMock


class TestDictionaryGenerator(unittest.TestCase):

    def get_suggested_name(self, dataset_name, modality_name):
        return f'read_dictionary:{dataset_name}->{modality_name}'

    def test_read_suggested(self):
        mn = 'test_suggested'
        fm = FileManagerMock(
            fake_values={
                FileManagerMock.get_dictionary_name(
                    dataset_name='a',
                    modality_name=mn,
                ):
                build_empty_dictionary(['a', 'b', 'e']),
                FileManagerMock.get_dictionary_name(
                    dataset_name='b',
                    modality_name=mn,
                ):
                build_empty_dictionary(['a', 'b', 'c', 'd'])
            })
        dg = Dictionary_Generator()

        dg.append_suggested_dictionary(dataset_name='a', modality_name=mn, FMSingleton=fm)\
            .append_suggested_dictionary(dataset_name='b', modality_name=mn, FMSingleton=fm)

        td = dg.get_merged_suggested_dictionary(modality_name=mn)
        npt.assert_equal(td['label'], np.linspace(0, 4, 5, dtype=int))
        npt.assert_equal(td['name'], np.array(['a', 'b', 'c', 'd', 'e']))

    def test_string_values(self):
        dg = Dictionary_Generator()

        mn = 'test_str_values'
        dg.append_values(modality_name=mn, values=pd.Series(['a', 'b', 'c']))\
            .append_values(modality_name=mn, values=pd.Series(['a', 'b', 'd']))

        td = dg.get(modality_name=mn)
        npt.assert_equal(td['label'], np.linspace(0, 3, 4, dtype=int))
        npt.assert_equal(td['name'], np.array(['a', 'b', 'c', 'd']))

    def test_boolean_values(self):
        dg = Dictionary_Generator()

        mn = 'test_bool_values'
        dg.append_values(modality_name=mn, values=pd.Series([True, False, True]))

        td = dg.get(modality_name=mn)
        npt.assert_equal(td['label'], np.linspace(0, 1, 2, dtype=int))
        npt.assert_equal(td['name'], np.array([True, False]))

    def test_values_not_in_suggested(self):
        mn = 'test_not_in_suggested'
        fm = FileManagerMock(
            fake_values={
                FileManagerMock.get_dictionary_name(
                    dataset_name='a',
                    modality_name=mn,
                ):
                build_empty_dictionary(['a', 'b', 'e']),
                FileManagerMock.get_dictionary_name(
                    dataset_name='b',
                    modality_name=mn,
                ):
                build_empty_dictionary(['a', 'b', 'c', 'd'])
            })
        dg = Dictionary_Generator()

        dg.append_suggested_dictionary(dataset_name='a', modality_name=mn, FMSingleton=fm)\
            .append_suggested_dictionary(dataset_name='b', modality_name=mn, FMSingleton=fm)

        dg.append_values(modality_name=mn, values=pd.Series(['a', 'b', 'c']))\
            .append_values(modality_name=mn, values=pd.Series(['a', 'b', 'd', 'i']))

        self.assertRaises(IndexError, lambda: dg.get(modality_name=mn))

        td = dg.get(modality_name=mn, action_on_missing='silent')
        npt.assert_equal(td['label'], np.linspace(0, 4, 5, dtype=int))
        npt.assert_equal(td['name'], np.array(['a', 'b', 'c', 'd', 'e']))


if __name__ == '__main__':
    unittest.main()
