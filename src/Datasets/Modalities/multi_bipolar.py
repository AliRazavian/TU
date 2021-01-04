import numbers
import numpy as np
import pandas as pd
from file_manager import File_Manager
from .bipolar import Bipolar, fix_dims


class Multi_Bipolar(Bipolar):

    def report_modality_specific_epoch_summary(self, summary):
        all_targets = self.labels.values
        if not self.get_cfgs('to_each_view_its_own_label'):
            all_targets = all_targets[self.content.index.get_level_values(1) == 0]

        for idx, (csv_column, name) in enumerate(self.column_map.items()):
            outputs = self.get_runtime_value('output', csv_column_name=csv_column, convert_to_values=True)

            pseudo_outputs = None
            if 'pseudo_output' in self.runtime_values and csv_column in self.runtime_values['pseudo_output']:
                pseudo_outputs = self.get_runtime_value(
                    'pseudo_output',
                    csv_column_name=csv_column,
                    convert_to_values=True,
                )

            merged_modality_name = f'{self.get_name()}_{name}'
            targets = all_targets[:, idx]
            performance = self.compute_performance(outputs, targets, prefix='')
            summary['modalities'][merged_modality_name].update(performance)

            if pseudo_outputs is None:
                continue

            performance = self.compute_performance(pseudo_outputs, targets, prefix='pseudo_')
            summary['modalities'][merged_modality_name].update(performance)

    def analyze_modality_specific_results(self, batch):
        results = {}
        all_outputs = self.unwrap(batch['results'][self.get_name()].pop('output'))
        all_targets = self.unwrap(batch['results'][self.get_name()].pop('target'))
        all_pseudo_outputs = None
        if 'pseudo_output' in batch['results'][self.get_name()]:
            all_pseudo_outputs = self.unwrap(batch['results'][self.get_name()].pop('pseudo_output'))

        for idx, (csv_column, name) in enumerate(self.column_map.items()):
            outputs = all_outputs[:, idx]
            targets = all_targets[:, idx]

            results = self.analyze_modality_accuracy(
                output=outputs,
                target=targets,
                indices=batch['indices'],
                results=results,
                prefix='',
                subgroup_name=csv_column,
            )

            if all_pseudo_outputs is not None:
                pseudo_outputs = all_pseudo_outputs[:, idx]
                results = self.analyze_modality_accuracy(
                    output=pseudo_outputs,
                    target=targets,
                    indices=batch['indices'],
                    results=results,
                    prefix='pseudo_',
                    subgroup_name=csv_column,
                )

            full_name = f'{self.get_name()}_{name}'
            batch['results'][full_name].update(results)

    def get_item(self, index, num_view=None, spatial_transfrom=None):
        if self.to_each_view_its_own_label:
            labels = np.array(self.labels.loc[index, :])
        else:
            # All labels are assumed to be identical, hence [0]
            labels = np.array(self.labels.loc[index, 0])
            # Without this all the labels get concatenated into one dimension
            labels = labels.reshape(1, -1)

        return {self.get_batch_name(): labels}

    def prep_content(self):
        assert isinstance(self.content, pd.DataFrame), f'The content should be a dataframe for {self.get_name()}'
        assert len(self.content.columns) > 0, f'There are no columns in the dataframe for {self.get_name()}'
        assert len(self.content) > 0, f'The dataframe is empty for {self.get_name()}'

        self.content = self.content.apply(lambda series: series.apply(lambda x: x.lower()
                                                                      if (isinstance(x, str)) else x))

    def get_num_classes(self):
        return len(self.content.columns)

    def get_loss_type(self):
        return 'bipolar_margin_loss'

    def convert_class_names_to_indices(self):
        if self.dictionary is None:
            raise Exception(f'No dictionary has been generated for {self.get_name()}')

        self.labels = self.content.apply(
            lambda s: s.map(self.cls_name_to_label).fillna(self.ignore_index).astype(np.int32))
        assert len(self.labels.dtypes.unique()) == 1, 'Expected all label types to be of equal type'
        assert self.labels.dtypes[0].kind in ['i', 'u'], 'Expected integer (either signed or unsigned)'

        return self

    @staticmethod
    def get_column_2_name_map(column_defintions, modality_name):
        if column_defintions is None:
            raise ValueError(f'The modality {modality_name} should have a columns attribute')

        columns = {}
        for column in column_defintions:
            if isinstance(column, dict):
                if 'name' not in column:
                    raise IndexError(f'The modality {modality_name} has a dictionary without the entry "name"')
                if 'csv_name' not in column:
                    raise IndexError(f'The modality {modality_name} has a dictionary without the entry "csv_name"')
                columns[column['csv_name']] = column['name']
            elif isinstance(column, str):
                columns[column] = column
            else:
                raise ValueError(f'The column defintions {modality_name} can only be string or dict')

        return columns

    @staticmethod
    def get_csv_column_names(column_defintions, modality_name):
        return Multi_Bipolar.get_column_2_name_map(column_defintions=column_defintions,
                                                   modality_name=modality_name).keys()

    @property
    def column_map(self):
        return Multi_Bipolar.get_column_2_name_map(column_defintions=self.get_cfgs('columns'),
                                                   modality_name=self.get_name())

    @property
    def csv_columns(self):
        return self.column_map.keys()

    def get_csv_column_name(self, name: str):
        for csv, colum_name in self.column_map.items():
            if colum_name == name:
                return csv
        raise IndexError(f'There is no column named {name} in {str(self.column_map)}')

    def collect_statistics(self):
        for column in self.csv_columns:
            column_stats = self.get_content_statistics(labels=self.labels[column])
            self.label_stats[column] = column_stats

    def get_loss_weight(self):
        loss_weights = np.array([s['loss_weight'] for s in self.label_stats.values()], dtype=np.float32)
        return self.wrap(loss_weights)

    def get_initial_runtime_value(self, runtime_value_name: str):
        runtime_value_name = runtime_value_name.lower()
        assert runtime_value_name not in self.runtime_values,\
            f'Trying to init {runtime_value_name} but it is already initialized in {self.get_name()}'

        runtime_value = self.content.copy()

        for column in self.csv_columns:
            runtime_value[column].name = self.get_runtime_name(column=column, runtime_value_name=runtime_value_name)
            default_value = self.get_default_value(runtime_value_name=runtime_value_name, csv_column_name=column)
            if isinstance(default_value, (numbers.Number, bool, str)):
                runtime_value[column].values.fill(default_value)
            elif isinstance(default_value, dict):
                runtime_value[column] = runtime_value[column].map(default_value)

        if self.get_cfgs('to_each_view_its_own_label'):
            return runtime_value

        numpy_runtime_values = {}
        for column in self.csv_columns:
            rv = runtime_value[column]
            rv = rv[rv.index.get_level_values(1) == 0]
            numpy_runtime_values[column] = rv.to_numpy()

        return numpy_runtime_values

    def get_runtime_name(self, runtime_value_name, column):
        return f'{self.get_name()}_{column}_{runtime_value_name}'

    def get_default_value(self, runtime_value_name, csv_column_name):
        default_value = 0.
        if runtime_value_name == 'informativeness':
            default_value = self.get_initial_informativeness(csv_column_name=csv_column_name)
        elif runtime_value_name in [
                'entropy',
                'accuracy',
                'output',
                'pseudo_entropy',
                'pseudo_accuracy',
                'pseudo_output',
        ]:
            default_value = self.get_cfgs('ignore_index', default=-100.)
        elif runtime_value_name in ['prediction', 'pseudo_prediction']:
            default_value = ''
        else:
            raise BaseException(f'Unknown runtime {runtime_value_name}')

        return default_value

    def get_initial_informativeness(self, csv_column_name: str):
        if csv_column_name not in self.label_stats:
            raise Exception(f'The stats has not been fetched for {self.get_name()}::{csv_column_name}')

        label_informativeness = {}
        for label_name, label_index in self.cls_name_to_label.items():
            label_informativeness[label_name] = self.label_stats[csv_column_name]['label_informativeness'][label_index]

        return label_informativeness

    def get_runtime_value(self, runtime_value_name, csv_column_name, convert_to_values=False):
        """
        runtime values are the values that are computed during
        the runtime, like entropy, accuracy, etc...
        We store them during training and testing to be able to
        measure performance, informativeness.
        """
        name = runtime_value_name.lower()

        if name not in self.runtime_values:
            self.runtime_values[name] = self.get_initial_runtime_value(runtime_value_name=name)

        if csv_column_name not in self.runtime_values[name]:
            raise IndexError(f'The {self.get_name()} failed to init runtime_values for {csv_column_name}' +
                             f', not among {str(self.runtime_values[name].keys())}')

        rv = self.runtime_values[name][csv_column_name]
        if convert_to_values and isinstance(rv, pd.Series):
            rv = rv.values
        return rv

    def get_runtime_values(self):
        if self.get_cfgs('to_each_view_its_own_label'):
            values = []
            for rt_dataframe in self.runtime_values.values():
                [values.append(rt_dataframe[c]) for c in rt_dataframe]

            return values

        # This logic is for mapping to the right subindices, i.e. copy onto
        values = []
        for runtime_value_name, rt_dict in self.runtime_values.items():
            for column, numpy_data in rt_dict.items():
                series = pd.Series(numpy_data)
                series.name = self.get_runtime_name(runtime_value_name=runtime_value_name, column=column)
                values.append(series)

        df_raw = pd.concat(values, axis=1)
        df_w_index = self.content.copy()
        df_w_index['__empty__'] = np.nan
        df_w_index = df_w_index[['__empty__']]
        df_w_index = df_w_index.join(df_raw, on='index')
        df_w_index = df_w_index.drop(columns=['__empty__'])

        return [df_w_index[c] for c in df_w_index]

    def set_runtime_value(self, runtime_value_name, value, indices, subgroup_name):
        runtime_value = self.get_runtime_value(runtime_value_name, csv_column_name=subgroup_name)
        value = fix_dims(value)
        for i in range(len(indices)):
            runtime_value[indices[i]] = value[i]

    def init_dictionary(self):
        if self.get_cfgs('skip_dictionary_save', default=False):
            return

        column_dictionary = pd.DataFrame({
            'columns': [c for c in self.csv_columns],
            'labels': [self.column_map[c] for c in self.csv_columns],
            'index': range(len(self.csv_columns))
        })

        File_Manager().write_dictionary2logdir(dictionary=column_dictionary, modality_name=self.get_name())
