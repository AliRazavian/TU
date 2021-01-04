import time
import pandas as pd

from UIs.console_UI import Console_UI
from Graphs.graph import Graph
from Tasks.base_task import Base_Task
from GeneralHelpers import ProgressBar
from Datasets.dataset_factory import Dataset_Factory


def flatten_dict(dd, separator='_', prefix=''):
    return {
        prefix + separator + k if prefix else k: v for kk, vv in dd.items()
        for k, v in flatten_dict(vv, separator, kk).items()
    } if isinstance(dd, dict) else {
        prefix: dd
    }


class Training_Task(Base_Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(self.get_cfgs('task_type').lower() == 'training'.lower()),\
            "training task %s created for a non-training scenario:%s" % (
                self.get_name(), self.scenario_name)

        self.dataset_name = self.get_cfgs('dataset_name')
        self.graph_name = self.get_cfgs('graph_name')

        # Datasets used
        self.train_set_name = self.get_cfgs('train_set_name')
        if self.has_pseudo_labels:
            self.train_set_name = self.get_cfgs('pseudo_set_name', default=self.train_set_name)
        self.val_set_name = self.get_cfgs('val_set_name')
        self.test_set_name = self.get_cfgs('test_set_name')

        self.pi_model = self.get_cfgs('pi_model')
        self.validate_when_epoch_is_devisable_by = self.get_cfgs('validate_when_epoch_is_devisable_by')

        self.dataset = Dataset_Factory().get_dataset(
            dataset_name=self.dataset_name,
            batch_size_multiplier=self.get_cfgs('batch_size_multiplier', 1),
        )

        experiment_names = [self.train_set_name, self.val_set_name]
        if self.test_set_name:
            experiment_names.append(self.test_set_name)

        for experiment_name in experiment_names:
            # Make sure that we have all the expected datasets
            if experiment_name not in self.dataset.experiments:
                raise ValueError(f'The set \'{experiment_name}\' cannot be found in dataset "{self.dataset_name}"')

            self.graphs[experiment_name.lower()] = \
                Graph(graph_name=self.graph_name,
                      experiment_set=self.dataset.experiments[experiment_name],
                      task_cfgs=self.task_cfgs,
                      scene_cfgs=self.scene_cfgs,
                      scenario_cfgs=self.scenario_cfgs)

    def __len__(self):
        return len(self.dataset.experiments[self.train_set_name])

    def step(
            self,
            iteration_counter: int,
            scene_name: str,
    ):
        # Due to the parallel nature of loading the data we need to reset
        # the start time for the batch in order to get the true processing
        start_time = time.time()

        experiment_set = self.dataset.experiments[self.train_set_name]
        batch = next(experiment_set)
        if batch is None:
            self.end_epoch(experiment_name=self.train_set_name, scene_name=scene_name)
            if ((self.epoch < 5 and iteration_counter < 1e3)
                    or self.epoch % self.validate_when_epoch_is_devisable_by == 0):
                self.validate(iteration_counter, scene_name=scene_name)
            batch = next(experiment_set)
            if batch is None:
                raise Exception('The next batch after resetting was empty!?')

        batch['time']['start'] = start_time
        batch.update({
            'epoch': self.epoch,
            'graph_name': self.graphs[self.train_set_name.lower()].get_name(),
            'task_name': self.get_name(),
            'iteration_counter': iteration_counter,
        })
        success = self.graphs[self.train_set_name.lower()].train(batch)
        if not success:
            return False

        Console_UI().add_batch_results(batch)
        return True

    def validate(
            self,
            iteration_counter,
            scene_name: str,
            set_name=None,
    ):
        if set_name is None:
            set_name = self.val_set_name

        if set_name not in self.dataset.experiments:
            raise ValueError(f'The set "{set_name}" cannot be found in data')

        experiment_set = self.dataset.experiments[set_name]
        Console_UI().inform_user(f'Validating {self.get_name()}: {set_name}')
        bar = ProgressBar(total=len(experiment_set))
        for batch in experiment_set:
            if batch is None:
                bar.done()
                break

            bar.current += 1
            bar()

            batch.update({
                'epoch': self.epoch,
                # 'graph_name': self.graphs[self.train_set_name.lower()].get_name(),
                'graph_name': self.graphs[set_name.lower()].get_name(),
                'task_name': self.get_name(),
                'iteration_counter': iteration_counter,
            })
            self.graphs[set_name].eval(batch)
        self.end_epoch(set_name, scene_name=scene_name)

    def test(self, iteration_counter, scene_name):
        if not self.test_set_name:
            return None

        return self.validate(iteration_counter=iteration_counter, scene_name=scene_name, set_name=self.test_set_name)

    def end_epoch(self, experiment_name: str, scene_name: str):
        experiment_set = self.dataset.experiments[experiment_name]
        summary = {'epoch': self.epoch, 'graph_name': self.graph_name, 'task_name': self.get_name()}

        if experiment_name == self.train_set_name:
            summary['epoch_size'] = len(experiment_set)
            self.save(scene_name='last')
            self.epoch += 1

        experiment_set.end_epoch(summary=summary, scene_name=scene_name)
        Console_UI().add_epoch_results(summary)

    def update_learning_rate(self, learning_rate):
        self.graphs[self.train_set_name].update_learning_rate(learning_rate)
        self.graphs[self.val_set_name].update_learning_rate(learning_rate)
        if self.test_set_name:
            self.graphs[self.test_set_name].update_learning_rate(learning_rate)

    def save(self, scene_name='last'):
        self.graphs[self.train_set_name].save(scene_name)

    def stochastic_weight_average(self):
        experiment_set = self.dataset.experiments[self.train_set_name]
        set_name = self.train_set_name.lower()
        has_run_average = self.graphs[set_name].update_stochastic_weighted_average_parameters()  # noqa: F841
        # Think this through - we can probably skip this step but it doesn't harm anything
        # if not has_run_average:
        #     return False

        self.graphs[set_name].prepare_for_batchnorm_update()
        self.graphs[set_name].train()
        experiment_set.reset_epoch()
        while True:
            Console_UI().inform_user('==>updating batchnorm')
            i = 0
            for batch in experiment_set:
                i += 1
                if i % 100 == 1:
                    Console_UI().inform_user(
                        f'Updating batchnorm for {self.get_name()}, doing {self.train_set_name} on step {i}')

                if batch is None:
                    self.graphs[set_name].finish_batchnorm_update()
                    return

                self.graphs[set_name].update_batchnorm(batch)

    def get_memory_usage_profile(self):
        usage = {}

        def merge_memory(d1, d2):
            if d2 is None:
                return d1
            for key in ['param', 'total']:
                if d1[key] is None:
                    d1[key] = d2[key]
                elif d2[key] is not None:
                    d1[key] += d2[key]
            return d1

        for model_name in self.graphs[self.train_set_name].models.keys():
            encoder_memory = {'param': 0, 'total': 0}
            model = self.graphs[self.train_set_name].models[model_name]
            if model.encoder is not None:
                if hasattr(model.encoder, 'network_memory_usage'):
                    merge_memory(encoder_memory, model.encoder.network_memory_usage)
                elif hasattr(model.encoder, 'neural_nets'):
                    for n in model.encoder.neural_nets:
                        merge_memory(encoder_memory, n.network_memory_usage)

            decoder_memory = {'param': 0, 'total': 0}
            if model.decoder is not None:
                if hasattr(model.decoder, 'network_memory_usage'):
                    merge_memory(decoder_memory, model.decoder.network_memory_usage)
                elif hasattr(model.decoder, 'neural_nets'):
                    for n in model.decoder.neural_nets:
                        merge_memory(decoder_memory, n.network_memory_usage)

            usage[model_name] = {'encoder': encoder_memory, 'decoder': decoder_memory}

        usage_dataset = pd.DataFrame({k: flatten_dict(usage[k]) for k in usage.keys()}).transpose()
        usage_dataset['total'] = usage_dataset['encoder_total'] + usage_dataset['decoder_total']
        usage_dataset['param_total'] = usage_dataset['encoder_param'] + usage_dataset['decoder_param']
        usage_dataset = usage_dataset.sort_values(by="total", ascending=False)

        return usage_dataset
