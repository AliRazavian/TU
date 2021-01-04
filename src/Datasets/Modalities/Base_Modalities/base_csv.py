import numbers
import pandas as pd

from abc import ABCMeta, abstractmethod

from .base_explicit import Base_Explicit


class Base_CSV(Base_Explicit, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_view = None

        assert('content' in kwargs and
               not kwargs['content'] is None and
               (isinstance(kwargs['content'], pd.Series) or
                isinstance(kwargs['content'], pd.DataFrame))),\
            'A CSV modality should either have a pd.Series, pd.DataFrame or None as content'

        self.dataset_name = kwargs.pop('dataset_name')
        self.content = kwargs.pop('content')

    def is_csv(self):
        return True

    def get_default_value(self, runtime_value_name):
        if runtime_value_name == 'informativeness':
            default_value = self.get_initial_informativeness()
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
            raise BaseException('Unknown runtime %s' % (runtime_value_name))

        return default_value

    def get_runtime_value(self, runtime_value_name: str, convert_to_values=False):
        """
        runtime values are the values that are computed during
        the runtime, like entropy, accuracy, etc...
        We store them during training and testing to be able to
        measure performance, informativeness.
        """
        name = runtime_value_name.lower()

        if name not in self.runtime_values:
            self.runtime_values[name] = self.get_initial_runtime_value(runtime_value_name=name)

        rv = self.runtime_values[name]
        if convert_to_values and isinstance(rv, pd.Series):
            rv = rv.values
        return rv

    def get_initial_informativeness(self):
        return 1.

    def get_initial_runtime_value(self, runtime_value_name: str):
        runtime_value_name = runtime_value_name.lower()

        default_value = self.get_default_value(runtime_value_name=runtime_value_name)
        assert runtime_value_name not in self.runtime_values, \
            f'Trying to init {runtime_value_name} but it is already initialized in {self.get_name()}'

        runtime_value = self.content.copy()
        runtime_value.name = f'{self.get_name()}_{runtime_value_name}'

        if isinstance(default_value, (numbers.Number, bool, str)):
            runtime_value.values.fill(default_value)
        elif isinstance(default_value, dict):
            runtime_value = runtime_value.map(default_value)
        return runtime_value

    @abstractmethod
    def set_runtime_value(
            self,
            runtime_value_name,
            value,
            indices,
            sub_indices,
    ):
        pass

    def get_sub_content(self, index, sub_index):
        content = 'not found!'
        try:
            content = self.content[index][sub_index]
        except KeyError:
            raise KeyError("Could not locate [{index}][{sub_index}] in the '{dataset}' dataset".format(
                index=index,
                sub_index=sub_index,
                dataset=self.dataset_name,
            ))
        return content

    def get_content(self, index):
        return [self.get_sub_content(index, i) for i in range(len(self.content[index]))]

    # Not used prior to name change - left in case we need it :-)
    # def previous_set_runtime_value(
    #         self,
    #         runtime_value_name,
    #         runtime_value_series,
    # ):
    #     assert isinstance(runtime_value_series, pd.Series), \
    #         f'In {self.get_name()}, {runtime_value_name} should be a pandas series.' + \
    #         f' Instead, found {str(type(runtime_value_series))}'

    #     name = runtime_value_name.lower()
    #     self.runtime_values[name] = runtime_value_series
    #     if (isinstance(runtime_value_series[0], str)):
    #         self.runtime_values[name] = self.runtime_values[name].apply(ast.literal_eval)
