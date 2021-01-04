import numpy as np

from .Base_Modalities.base_value import Base_Value
from .Base_Modalities.base_output import Base_Output


class Multi_Coordinate(Base_Value, Base_Output):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels = len(self.content.columns)

    def get_num_classes(self):
        return self.num_channels

    def get_item(self, index, num_view=None, spatial_transfrom=None):

        values = np.array(self.content.loc[index, :])
        values = values.reshape(num_view, 1, self.num_channels // 2, 2)
        values = np.repeat(values, self.num_jitters, axis=1)
        return {self.get_batch_name(): values}

    def analyze_modality_specific_results(self, batch):
        predictions = batch['predicted_' + self.get_encoder_name()]
        predictions = predictions.mean(axis=1)
        predictions = predictions.reshape(predictions.shape[0], -1)

        for i, k in enumerate(self.content.columns):
            self.set_runtime_value('predicted_' + k, predictions[:, i], batch['indices'], batch['num_views'])

    def report_modality_specific_epoch_summary(self, summary):
        pass

    def report_runtime_value(self, runtime_value_name, value, indices, num_views):
        pass

    def has_regression_loss(self):
        return True

    def has_pseudo_label(self):
        # TODO: pseudo labels should probably exist for coordinates as well, the text may contain information
        # about angles that could help positioning
        return False

    def get_loss_type(self):
        return 'mse_with_spatial_transform'

    def set_runtime_value(self, runtime_value_name, value, indices, num_views):
        runtime_value = self.get_runtime_value(runtime_value_name)
        c = np.cumsum([0, *num_views])
        for i in range(len(indices)):
            runtime_value[indices[i]] = value[c[i]:c[i + 1]]

    def collect_statistics(self):
        self.label_stats['std'] = self.content.values.std()
        self.label_stats['max'] = self.content.values.max()
        self.label_stats['min'] = self.content.values.min()

    def get_default_value(self, runtime_value_name):
        return self.get_cfgs('ignore_index', default=-100.)

    def get_initial_runtime_value(self, runtime_value_name: str):
        runtime_value_name = runtime_value_name.lower()

        default_value = self.get_default_value(runtime_value_name=runtime_value_name)
        assert runtime_value_name not in self.runtime_values, \
            f'Trying to init {runtime_value_name} but it is already initialized in {self.get_name()}'

        column_name = None
        for prefix in self.modality_cfgs['column_prefixes']:
            for suffix in ['_x', '_y']:
                column_suggestion = f'{prefix}{suffix}'
                chop_chars = len(runtime_value_name) - len(column_suggestion)
                if chop_chars >= 0:
                    if runtime_value_name[chop_chars:] == column_suggestion.lower():
                        column_name = column_suggestion
                        break
            if column_name is not None:
                break

        assert column_name is not None, f'Failed to identify corresponding column to {runtime_value_name}'
        cns = ", ".join(self.content.columns)
        assert column_name in self.content, f'Could not find {column_name} among the columns: {cns}'

        runtime_value = self.content[column_name].copy()
        runtime_value.name = f'{self.get_name()}_{runtime_value_name}'

        runtime_value.values.fill(default_value)
        return runtime_value
