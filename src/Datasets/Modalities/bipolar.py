import numpy as np
import torch
from collections import Counter

from sklearn.metrics import roc_auc_score

from .Base_Modalities.base_label import Base_Label
from .Base_Modalities.base_output import Base_Output
from UIs.console_UI import Console_UI
from global_cfgs import Global_Cfgs


# Helpers for dim bug
def check_dims(var):
    if (not isinstance(var, torch.Tensor) and not isinstance(var, np.ndarray)):
        return False

    if var.shape == ():
        return False

    return True


def fix_dims(var):
    if check_dims(var):
        return var
    return np.array([var])


def compute_auc(outputs, targets):
    mask = np.logical_or(targets == 1, targets == -1)
    if len(mask) < 2 or mask.sum() < 2:
        return np.nan

    try:
        auc = roc_auc_score(y_true=targets[mask] == 1, y_score=outputs[mask])
    except IndexError as error:
        # TODO: Why is this throwing?
        msg = f'IndexError in AUC calculation: {error}'
        Console_UI().warn_user(msg)
        return np.nan
    except ValueError as error:
        msg = f'ValueError in AUC calculation: {error}'
        Console_UI().warn_user(msg)
        return np.nan

    return auc


class Bipolar(Base_Label, Base_Output):

    def compute_performance(self, outputs, targets, prefix: str):
        outputs = fix_dims(outputs)
        targets = fix_dims(targets)

        positive_mask = targets == 1
        negative_mask = targets == -1
        true_positive = np.sum(outputs[positive_mask] > 0)
        false_positive = np.sum(outputs[negative_mask] > 0)
        false_negative = np.sum(outputs[positive_mask] < 0)
        true_negative = np.sum(outputs[negative_mask] < 0)

        sensitivity = np.nan
        specificity = np.nan
        precision = np.nan
        negative_predictive_value = np.nan
        auc = np.nan

        if np.any(positive_mask):
            sensitivity = true_positive / positive_mask.sum()
        if np.any(negative_mask):
            specificity = true_negative / negative_mask.sum()
        if true_positive + false_positive:
            precision = true_positive / (true_positive + false_positive)
        if true_negative + false_negative:
            negative_predictive_value = true_negative / (true_negative + false_negative)

        # AUC can't be calculated if there is only one group
        if np.any(positive_mask) and np.any(negative_mask):
            auc = compute_auc(outputs=outputs, targets=targets)

        accuracy = self.compute_accuracy(outputs, targets)
        accuracy = self.get_mean_accuracy(accuracy)
        # kappa = self.compute_kappa(accuracy)

        return {
            f'{prefix}sensitivity': sensitivity,
            f'{prefix}specificity': specificity,
            f'{prefix}precision': precision,
            f'{prefix}negative_predictive_value': negative_predictive_value,
            f'{prefix}auc': auc,
            f'{prefix}accuracy': accuracy,
            # f'{prefix}kappa': kappa, - not really using...
        }

    def report_modality_specific_epoch_summary(self, summary):
        outputs = self.get_runtime_value('output', convert_to_values=True)
        targets = self.labels.values

        performance = self.compute_performance(outputs, targets, prefix='')
        summary['modalities'][self.get_name()].update(performance)

        if 'pseudo_output' not in self.runtime_values:
            return
        pseudo_outputs = self.get_runtime_value('pseudo_output', convert_to_values=True)

        performance = self.compute_performance(pseudo_outputs, targets, prefix='pseudo_')
        summary['modalities'][self.get_name()].update(performance)

    def analyze_modality_accuracy(self, output, target, indices, results, prefix, subgroup_name=None):
        self.set_runtime_value(f'{prefix}output', value=output, indices=indices, subgroup_name=subgroup_name)

        entropy = self.compute_entropy(output=output)
        self.set_runtime_value(f'{prefix}entropy', value=entropy, indices=indices, subgroup_name=subgroup_name)

        accuracy = self.compute_accuracy(output=output, target=target)
        self.set_runtime_value(f'{prefix}accuracy', value=accuracy, indices=indices, subgroup_name=subgroup_name)
        results[f'{prefix}accuracy'] = self.get_mean_accuracy(accuracy)

        return results

    def analyze_modality_specific_results(self, batch):
        results = {}
        output = self.unwrap(batch['results'][self.get_name()].pop('output')).reshape(-1)
        target = self.unwrap(batch['results'][self.get_name()].pop('target')).reshape(-1)

        results = self.analyze_modality_accuracy(
            output=output,
            target=target,
            indices=batch['indices'],
            results=results,
            prefix='',
        )

        if 'pseudo_output' in batch['results'][self.get_name()]:
            pseudo_output = self.unwrap(batch['results'][self.get_name()].pop('pseudo_output')).squeeze()
            results = self.analyze_modality_accuracy(
                output=pseudo_output,
                target=target,
                indices=batch['indices'],
                results=results,
                prefix='pseudo_',
            )

        batch['results'][self.get_name()].update(results)

    def set_runtime_value(self, runtime_value_name, value, indices, subgroup_name=None):
        runtime_value = self.get_runtime_value(runtime_value_name)
        value = fix_dims(value)
        for i in range(len(indices)):
            runtime_value[indices[i]] = value[i]

    def compute_entropy(self, output):
        return np.exp(-(output**2))

    def compute_accuracy(self, output, target):
        # For some reason the dimension can get lost and we get only a single value
        # TODO: fix output dim lost
        output = fix_dims(output)
        target = fix_dims(target)

        accuracy = np.ones(target.shape) * self.ignore_index
        accuracy[target == 1] = output[target == 1] > 0
        accuracy[target == -1] = output[target == -1] < 0
        return accuracy

    # def compute_kappa(self, accuracy):
    #     return 1. - (1. - accuracy) / (1. - max(self.label_stats['label_likelihood']))

    def collect_statistics(self, labels=None):
        self.label_stats.update(self.get_content_statistics())

    def get_content_statistics(self, labels=None):
        if labels is None:
            labels = self.labels[self.labels != self.ignore_index]

        if not self.get_cfgs('to_each_view_its_own_label'):
            # labels = [l.to_list()[0] for idx, l in labels.groupby(level=0)]
            # Better performance:
            labels = labels[labels.index.get_level_values(level=1) == 0]

        statistics = {}
        c = Counter(labels)
        statistics['labels'] = np.array([-1, 0, 1])
        statistics['label_count'] = np.array([c[-1], c[0], c[1]])
        statistics['label_likelihood'] = statistics['label_count'] / np.sum(statistics['label_count'])

        loss_type = Global_Cfgs().get('loss_weight_type', None)
        if loss_type == 'max':
            # Limit the loss to max 1/2:th of the total when there are > 2000 observations
            # We've found that using np.sqr or smaller limitations causes strange collapses
            # in the multi-bipolar estimates
            total_with_label = sum([statistics['label_count'][i] for i in [0, 2]])
            max_for_loss = max(1000, total_with_label / 3)
            statistics['loss_weight'] = np.array([1 / min(c[i] + 1, max_for_loss) for i in [-1, 1]])
        elif loss_type == 'sqrt':
            # Square root loss
            statistics['loss_weight'] = np.array([1 / np.sqrt(c[i] + 1) for i in [-1, 1]])
        else:
            # Basic loss
            statistics['loss_weight'] = np.array([1 / (c[i] + 1) for i in [-1, 1]])

        # Normalize the weights so that sum equals 1
        statistics['loss_weight'] /= np.sum(statistics['loss_weight'])
        statistics['num_classes'] = self.get_num_classes()

        label_informativeness = {}
        for label_index, label_likelihood in zip(statistics['labels'], statistics['label_likelihood']):
            label_informativeness[label_index] = (1 - label_likelihood) * self.signal_to_noise_ratio

        statistics['label_informativeness'] = label_informativeness

        return statistics

    def get_num_classes(self):
        return 1

    def get_loss_type(self):
        return 'bipolar_margin_loss'
