import time
import unittest
import pandas as pd
import numpy as np
import numpy.testing as npt
from Datasets.Modalities.multi_coordinate import Multi_Coordinate
from Datasets.Modalities.image_from_filename import Image_from_Filename
from Utils.image_tools import convert_original_points_to_cropped_image
from Utils.image_tools import plot_points_on_image
from tests.helpers.ConfigMock import ConfigMock


class TestRegression(unittest.TestCase):

    def get_regression(self, content):
        return Multi_Coordinate(
            dataset_name='test_data',
            dataset_cfgs={},
            experiment_name='test_exp',
            experiment_cfgs={'num_jitters': 3},
            modality_name='test_modality',
            content=content,
            modality_cfgs={},
            global_cfgs=ConfigMock(),
            dictionary=None,
        )

    def get_image(self, content, modality_cfgs):

        return Image_from_Filename(
            dataset_name='test_data',
            dataset_cfgs={'test_data_root': 'test_data/imgs'},
            experiment_name='test_exp',
            experiment_cfgs={'num_jitters': 3},
            modality_name='test_modality',
            content=content,
            modality_cfgs=modality_cfgs,
            global_cfgs=ConfigMock(),
            dictionary=None,
        )

    def test_basic_instantiation(self):

        annotation = pd.read_csv('test_data/xray_test_wrist/small_sample_from_wrist_with_xy.csv')
        annotation.set_index(['index', 'sub_index'], inplace=True)

        image_modality = self.get_image(content=annotation['Filename'],
                                        modality_cfgs={
                                            'colorspace': 'gray',
                                            'spatial_transform': 'random',
                                            'scale_to': 512,
                                            'keep_aspect': True,
                                            'width': 256,
                                            'height': 320
                                        })
        reg_modality = self.get_regression(content=annotation[[
            'data_frontal_ulna_axis_c1_x', 'data_frontal_ulna_axis_c1_y', 'data_frontal_ulna_axis_c2_x',
            'data_frontal_ulna_axis_c2_y'
        ]])

        time_record = {'start': time.time(), 'load': {}, 'process': {}, 'encode': {}, 'decode': {}, 'backward': {}}
        batch = {'indices': [0, 1, 2], 'num_views': [3, 2, 3], 'time': time_record}

        batch.update(image_modality.get_batch(batch))
        batch.update(reg_modality.get_batch(batch))

        targets = batch['target_test_modality']
        images = batch['encoder_test_modality']
        M = batch['spatial_transforms']

        # TODO: Add more useful regression tests - this is only generated for creating a pass
        for i in range(targets.shape[0]):
            for j in range(targets.shape[1]):
                if not np.any(np.isnan(targets[i])):
                    transformed_targets = convert_original_points_to_cropped_image(
                        M=M[i, j],
                        original_points=targets[i][j],  # must be a numpy array of nx2 coordinates in relative values
                        original_im_size=batch['original_input_size'][i][j][0:2],
                        cropped_im_size=images[0].shape[2:])
                    npt.assert_array_equal(transformed_targets.shape, targets[i][j].shape)

        # plot_points_on_image()

        # print(targets.shape)
        # print(images.shape)
        # print(M.shape)


if __name__ == '__main__':
    unittest.main()
