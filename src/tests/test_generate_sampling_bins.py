import unittest
import pandas as pd
import numpy as np
from Datasets.helpers import generate_sampling_bins


class TestBinSampler(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        np.random.seed(3321)
        super().__init__(*args, **kwargs)
        self.annotations = pd.DataFrame(
            data={
                'num_views': [1, 1, 2, 2, 3, 3, 3, 1],
                'index': [0, 1, 2, 2, 3, 3, 3, 4],
                'sub_index': [0, 0, 0, 1, 0, 1, 2, 0],
            })
        self.annotations.set_index(['index', 'sub_index'], inplace=True)

    def test_batch_size_1(self):
        bins, bin_weight = generate_sampling_bins(
            annotations=self.annotations,
            batch_size=1,
            silent=True,
        )
        self.assertEqual(len(bins), 3)
        self.assertEqual(bin_weight, [1, 1, 1])

    def test_batch_size_smaller_than_dataset(self):
        bins, bin_weight = generate_sampling_bins(
            annotations=self.annotations,
            batch_size=2,
            silent=True,
        )
        self.assertEqual(len(bins), 3)
        self.assertEqual(bin_weight, [2, 2, 1])

    def test_batch_size_of_size_big_enough_for_max_views(self):
        bins, bin_weight = generate_sampling_bins(
            annotations=self.annotations,
            batch_size=self.annotations.num_views.max(),
            silent=True,
        )

        def flatten(l):
            return [item for sublist in l for item in sublist]

        self.assertEqual(len(bins), 3)
        self.assertEqual(len(flatten(bins)), 5)
        self.assertEqual(sum(bin_weight), 8)

    def test_batch_size_bigger_than_dataset(self):
        bins, bin_weight = generate_sampling_bins(
            annotations=self.annotations,
            batch_size=2000,
            silent=True,
        )
        self.assertEqual(len(bins), 1)
        self.assertEqual(bin_weight, [len(self.annotations)])

    def test_sorting_of_bins(self):
        big_annotations = pd.DataFrame(
            data={
                'num_views': [3, 3, 3, 2, 2, 2, 2, 2, 2],
                'index': [0, 0, 0, 1, 1, 2, 2, 3, 3],
                'sub_index': [0, 1, 2, 0, 1, 0, 1, 0, 1],
            })
        big_annotations.set_index(['index', 'sub_index'], inplace=True)

        np.random.seed(132)  # Gives [[0], [1, 2], [3]] & [3, 4, 2]
        bins, bin_weight = generate_sampling_bins(
            annotations=big_annotations,
            batch_size=4,
            silent=True,
        )
        self.assertEqual(bin_weight, [4, 3, 2])
        last = None
        for i in range(len(bins)):
            chosen_data = big_annotations.loc[bins[i]]
            self.assertEqual(len(chosen_data), bin_weight[i])
            if last is not None:
                self.assertLessEqual(len(chosen_data), last)
            last = len(chosen_data)

    def test_prioritize_many_bins(self):
        big_annotations = pd.DataFrame(
            data={
                'num_views': [4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 1],
                'index': [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4],
                'sub_index': [0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 2, 3, 0],
            })
        big_annotations.set_index(['index', 'sub_index'], inplace=True)

        np.random.seed(3123)  # Gives [[0], [1, 2], [3]] & [3, 4, 2]
        bins, bin_weight = generate_sampling_bins(
            annotations=big_annotations,
            batch_size=4,
            silent=True,
        )
        self.assertEqual(bin_weight[-1], 1)
        self.assertEqual(bin_weight[1], 4)
        self.assertEqual([len(bin) for bin in bins], [2, 1, 1, 1])
        last = None
        for i in range(len(bins)):
            chosen_data = big_annotations.loc[bins[i]]
            self.assertEqual(len(chosen_data), bin_weight[i])
            if last is not None:
                self.assertLessEqual(len(chosen_data), last)
            last = len(chosen_data)


if __name__ == '__main__':
    unittest.main()
