import traceback
import numpy as np
from abc import ABCMeta, abstractmethod

from global_cfgs import Global_Cfgs
from .pytorch_wrapper import wrap, unwrap
from UIs.console_UI import Console_UI


class Base_Modality(metaclass=ABCMeta):

    def is_explicit_modality(self):
        # this function will be overridden in Base_Explicit
        return False

    def is_implicit_modality(self):
        # this function will be overridden in Implicit
        return False

    def is_input_modality(self):
        # this function will be overridden in Base_Input
        return False

    def is_output_modality(self):
        # this function will be overridden in Base_Output
        return False

    def __init__(
            self,
            dataset_name: str,
            dataset_cfgs: dict,
            experiment_name: str,
            experiment_cfgs: dict,
            modality_name: str,
            modality_cfgs: dict,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_cfgs = dataset_cfgs
        self.experiment_name = experiment_name
        self.experiment_cfgs = experiment_cfgs
        self.modality_name = modality_name.lower()
        self.modality_cfgs = modality_cfgs
        self.consistency: str = None
        self.runtime_values = {}
        self.batch_loss = []

        self.num_jitters = self.get_cfgs('num_jitters', default=1)

        if ('tensor_shape' in self.modality_cfgs):
            self.set_tensor_shape(self.modality_cfgs['tensor_shape'])

    def get_runtime_values(self):
        return self.runtime_values.values()

    def has_reconstruction_loss(self):
        return False

    def has_identification_loss(self):
        return False

    def has_discriminator_loss(self):
        return False

    def has_classification_loss(self):
        return False

    def has_regression_loss(self):
        return False

    def analyze_results(self, batch, loss):
        self.batch_loss.append(loss)
        try:
            self.analyze_modality_specific_results(batch)
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            Console_UI().warn_user(f'Failed to get results for {self.modality_name}: {e}')
            pass

    def analyze_modality_specific_results(self, batch):
        pass

    def report_epoch_summary(self, summary):
        summary['modalities'][self.get_name()] = {}
        if len(self.batch_loss) > 0:
            summary['modalities'][self.get_name()]['loss'] = np.mean(self.batch_loss)
            self.batch_loss = []

        self.report_modality_specific_epoch_summary(summary)

    def report_modality_specific_epoch_summary(self, summary):
        pass

    def get_name(self):
        return self.modality_name.lower()

    def get_encoder_name(self):
        return 'encoder_%s' % (self.get_name())

    def get_decoder_name(self):
        return 'decoder_%s' % (self.get_name())

    def get_real_name(self):
        return self.get_batch_name()

    def get_fake_name(self):
        return self.get_decoder_name()

    def get_batch_name(self):
        return self.get_encoder_name()

    def get_target_name(self):
        return self.get_batch_name()

    def get_cfgs(self, name, default=None):
        user_cfgs = Global_Cfgs().get(name, default=None)
        if user_cfgs is not None:
            return user_cfgs
        if name in self.modality_cfgs:
            return self.modality_cfgs[name]
        if name in self.experiment_cfgs:
            return self.experiment_cfgs[name]
        if name in self.dataset_cfgs:
            return self.dataset_cfgs[name]
        return default

    def get_modality_cfgs(self):
        self.modality_cfgs['tensor_shape'] = self.get_tensor_shape()
        self.modality_cfgs['encoder_name'] = self.get_encoder_name()
        self.modality_cfgs['decoder_name'] = self.get_decoder_name()
        self.modality_cfgs['batch_name'] = self.get_batch_name()
        self.modality_cfgs['target_name'] = self.get_target_name()
        self.modality_cfgs['has_reconstruction_loss'] = self.has_reconstruction_loss()
        self.modality_cfgs['has_classification_loss'] = self.has_classification_loss()
        self.modality_cfgs['has_identification_loss'] = self.has_identification_loss()
        self.modality_cfgs['has_discriminator_loss'] = self.has_discriminator_loss()

        return self.modality_cfgs

    def get_shape(self):
        return [self.num_jitters, *self.get_tensor_shape()]

    def set_tensor_shape(self, shape):
        if len(shape) == 1:
            self.set_channels(shape[0])
        if len(shape) == 2:
            self.set_channels(shape[0])
            self.set_width(shape[1])
        if len(shape) == 3:
            self.set_channels(shape[0])
            self.set_width(shape[1])
            self.set_height(shape[2])
        if len(shape) == 4:
            self.set_channels(shape[0])
            self.set_width(shape[1])
            self.set_height(shape[2])
            self.set_depth(shape[3])

    def get_tensor_volume(self):
        return np.prod(self.get_tensor_shape())

    def get_shape_str(self):
        return 'x'.join(str(d) for d in self.get_shape())

    def set_consistency(self, consistency):
        self.consistency = consistency
        self.modality_cfgs['consistency'] = consistency

    def get_consistency(self):
        return self.consistency

    @abstractmethod
    def get_tensor_shape(self):
        """
        this function will be overridden in
        Base_Number, Base_Sequence, Base_Plane and Base_Volume"
        """
        pass

    def wrap(self, x):
        # TODO: We don't need the wrap as it is done by the dataloader
        return wrap(x)

    def unwrap(self, x):
        return unwrap(x)

    def is_classification(self):
        return False

    def is_regression(self):
        return False
