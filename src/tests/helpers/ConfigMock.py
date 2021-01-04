import os
from GeneralHelpers import Singleton
from global_cfgs import Global_Cfgs


class ConfigMock:

    def __init__(self, fake_values={}):
        init_values = {
            'DEVICE_BACKEND': 'gpu',
            'IMAGE_BACKEND': 'cv2',
            'output_name': 'test_output',
            'target_name': 'test_target',
            'ignore_index': -100,
        }
        init_values.update(fake_values)

        if Global_Cfgs in Singleton._instances:
            cfg = Global_Cfgs()
            cfg.cfgs = init_values
        else:
            Global_Cfgs(test_mode=True, test_init_values=init_values)

    def get(self, key, default=None):
        if (key in self.cfgs):
            return self.cfgs[key]

        return default
