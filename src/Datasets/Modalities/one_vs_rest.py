import numpy as np
from collections import Counter

from file_manager import File_Manager
from .Base_Modalities.base_label import Base_Label
from .Base_Modalities.base_output import Base_Output


class One_vs_Rest(Base_Label, Base_Output):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_to_cls_name = {}
        for name, label in self.cls_name_to_label.items():
            self.label_to_cls_name[label] = str(name).lower()

    def analyze_modality_specific_results(self, batch):
        results = {}

        output = self.unwrap(batch['results'][self.get_name()].pop('output'))
        # TODO: Ali'd response it could be related to the views - shouldn't be handled here
        output = output.reshape([-1, output.shape[-1]])

        target = self.unwrap(batch['results'][self.get_name()].pop('target')).reshape(-1)
        softmax = self.compute_softmax(output)

        entropy = self.compute_entropy(softmax=softmax)
        accuracy, prediction = self.compute_accuracy(output=output, target=target)

        tmp = [self.label_to_cls_name[int(i)] for i in prediction.reshape(-1)]
        prediction = np.array(tmp).reshape(prediction.shape)
        # TODO: Check prediction dtype

        self.set_runtime_value('accuracy', accuracy, batch['indices'], batch['num_views'])
        self.set_runtime_value('prediction', prediction, batch['indices'], batch['num_views'])
        self.set_runtime_value('entropy', entropy, batch['indices'], batch['num_views'])

        if ('pseudo_output' in batch['results'][self.get_name()]):
            pseudo_output = self.unwrap(batch['results'][self.get_name()].pop('pseudo_output'))
            pseudo_softmax = self.compute_softmax(pseudo_output)
            pseudo_entropy = self.compute_entropy(softmax=pseudo_softmax)
            pseudo_accuracy, pseudo_prediction = self.compute_accuracy(output=pseudo_output, target=target)
            pseudo_prediction = np.array([self.label_to_cls_name[int(i)]
                                          for i in pseudo_prediction.reshape(-1)])\
                                  .reshape(pseudo_prediction.shape)

            self.set_runtime_value('pseudo_accuracy', pseudo_accuracy, batch['indices'], batch['num_views'])
            self.set_runtime_value('pseudo_prediction', pseudo_prediction, batch['indices'], batch['num_views'])
            self.set_runtime_value('pseudo_entropy', pseudo_entropy, batch['indices'], batch['num_views'])
            results['pseudo_accuracy'] = self.get_mean_accuracy(pseudo_accuracy)

        results['accuracy'] = self.get_mean_accuracy(accuracy)
        batch['results'][self.get_name()].update(results)

    def report_modality_specific_epoch_summary(self, summary):
        accuracy = self.get_runtime_value('accuracy').values
        summary['modalities'][self.get_name()]['accuracy'] = self.get_mean_accuracy(accuracy)

        if 'pseudo_accuracy' not in self.runtime_values:
            return
        pseudo_accuracy = self.get_runtime_value('pseudo_accuracy').values
        summary['modalities'][self.get_name()]['pseudo_accuracy'] = self.get_mean_accuracy(pseudo_accuracy)

    def set_runtime_value(self, runtime_value_name, value, indices, num_views):
        runtime_value = self.get_runtime_value(runtime_value_name)

        if self.to_each_view_its_own_label:
            c = np.cumsum([0, *num_views])
            for i in range(len(indices)):
                runtime_value[indices[i]] = value[c[i]:c[i + 1]]
        else:
            for i in range(len(indices)):
                runtime_value[indices[i]] = value[i]

    def compute_softmax(self, output):
        output /= np.max(np.abs(output), axis=1)[:, np.newaxis]
        softmax = np.exp(output)
        softmax = softmax / (np.sum(softmax, axis=1)[:, np.newaxis])
        return softmax

    def compute_entropy(self, softmax):
        entropy = -np.sum(softmax * np.log(softmax + 1e-18), axis=1)
        return entropy

    def compute_accuracy(self, output, target):
        prediction = np.argmax(output, axis=1)
        accuracy = np.ones(target.shape, dtype='float32') * self.ignore_index
        target_valid_mask = target != self.ignore_index
        accuracy[target_valid_mask] = target[target_valid_mask] == prediction[target_valid_mask]

        return accuracy, prediction

    def collect_statistics(self):

        c = Counter(self.labels[self.labels != self.ignore_index])

        num_classes = self.get_num_classes()
        self.label_stats['num_classes'] = num_classes

        self.label_stats['labels'] = np.arange(num_classes)
        label_count_list = [c[i] for i in range(num_classes)]
        self.label_stats['label_count'] = np.array(label_count_list)
        self.label_stats['label_likelihood'] = self.label_stats['label_count'] / np.sum(self.label_stats['label_count'])
        # Could be replaced by log to reduce the imbalance
        loss_weight_list = [1 / np.sqrt(c[i] + 1) for i in range(num_classes)]
        self.label_stats['loss_weight'] = np.array(loss_weight_list, dtype='float32')
        self.label_stats['loss_weight'] /= np.sum(self.label_stats['loss_weight'])

        self.label_stats['label_informativeness'] = \
            (1 - self.label_stats['label_likelihood']) * self.signal_to_noise_ratio

    def get_num_classes(self):
        if self.dictionary is None:
            raise Exception(f'No dictionary has been initated for {self.get_name()}')

        return self.dictionary.label.max() + 1

    def get_loss_type(self):
        return 'cross_entropy'

    def init_dictionary(self):
        if self.dictionary is None:
            raise Exception(f'No dictionary has been initated for {self.get_name()}')

        File_Manager().write_dictionary2logdir(dictionary=self.dictionary, modality_name=self.get_name())
