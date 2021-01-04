import traceback
import math
import numpy as np
import time
import pandas as pd
from collections import defaultdict
from typing import Dict
from .helpers import get_modality_and_content, generate_sampling_bins
from .Modalities.Base_Modalities.pytorch_wrapper import collate_factory, convert_2_cuda
from torch.utils.data import DataLoader, Dataset
from file_manager import File_Manager
from global_cfgs import Global_Cfgs
from UIs.console_UI import Console_UI


class Experiment_Set(Dataset):

    def __init__(
            self,
            dataset_name: str,
            dataset_cfgs: Dict,
            experiment_name: str,
            experiment_cfgs: Dict,
            batch_size_multiplier: float,
    ):
        self.dataset_name = dataset_name
        self.dataset_cfgs = dataset_cfgs
        self.experiment_name = experiment_name
        self.experiment_cfgs = experiment_cfgs

        self.explicit_modalities = {}
        self.explicit_input_modalities = {}
        self.explicit_output_modalities = {}
        self.implicit_modalities = {}
        self.modalities = {}
        self.other_modalities = {}

        self.multi_view_per_sample = self.get_cfgs('multi_view_per_sample', default=True)
        self.batch_size = self.get_cfgs('batch_size')
        self.batch_size_multiplier = None
        self.set_batch_size_multiplier(number=batch_size_multiplier)

        self.setup_annotations()
        self.setup_modalities()
        self.reset_epoch()
        self.dataloader_iterator = None

    def set_batch_size_multiplier(self, number: float):
        assert number > 0, 'The batch size multiplier must be greater than 0 as we can\'t have negative sized batches'
        if self.batch_size_multiplier is None:
            self.batch_size_multiplier = number
        if self.batch_size_multiplier != number:
            # Rebuild the batches once we have a new batch size
            self.batch_size_multiplier = number
            self.reset_epoch()

    def __iter__(self):
        # When a for loop calls for the iterator it should be reset
        self.dataloader_iterator = None
        return self

    def __len__(self):
        return len(self.bins)

    def __next__(self):
        batch = None
        i = 0
        iterator = self.get_iterator()
        # start_time = time.time()
        while (batch is None and i < 5):
            try:
                batch = next(iterator)
                if batch['encoder_image'].max() == 0:
                    raise ValueError('No non-zero images in batch - check file folder')

                # Convert all tensors to cuda if environment calls for it
                for key in batch:
                    batch[key] = convert_2_cuda(batch[key])

                batch_size_info = f'{self.bin_weights[self.batch_index]} ({len(self.bins[self.batch_index])} bins)'
                batch.update({
                    'batch_index': self.batch_index,
                    'epoch_size': len(self),
                    'batch_size': batch_size_info,
                })
            except StopIteration:
                self.reset_epoch()
                return None
            except Exception as ex:
                batch = None
                Console_UI().warn_user(f'Failed to load batch: "{ex}"')
                traceback.print_exception(type(ex), ex, ex.__traceback__)

            self.batch_index += 1
            i += 1
            if i >= 5:
                raise Exception('Failed multiple times when trying to retrieve batch')

        # Check images - seem ok
        # import cv2

        # for i in range(batch['encoder_image'].shape[0]):
        #     for ii in range(batch['encoder_image'].shape[1]):
        #         img = batch['encoder_image'][i, ii, 0] * 255
        #         cv2.imwrite(f'/home/max/tmp/test{i}_{ii}.png', img.reshape(256, 256, 1).cpu().numpy())

        # Profile time for loading batch - not much is gained by more than two workers (time is 0.01 to 0.1 seconds)
        # print("Time spent time retrieving batch: %0.2f" % (time.time() - start_time))
        return batch

    def __getitem__(self, index):
        indices = self.bins[index]
        return self.get_batch(indices)

    def get_iterator(self):
        if self.dataloader_iterator is None:
            loader = DataLoader(
                dataset=self,
                batch_size=1,  # The batch size is decided when generating the bins
                shuffle=False,  # The bin generation shuffles
                num_workers=Global_Cfgs().get('num_workers'),
                collate_fn=collate_factory(keys_2_ignore=list(self.get_batch_defaults([0]).keys())),
            )
            self.dataloader_iterator = iter(loader)
        return self.dataloader_iterator

    def reset_epoch(self):
        self.batch_index = 0
        batch_size = math.ceil(self.batch_size * self.batch_size_multiplier)
        self.bins, self.bin_weights = generate_sampling_bins(
            annotations=self.annotations,
            batch_size=batch_size,
        )
        self.dataloader_iterator = None
        return self.bins

    def get_batch_defaults(self, indices):
        time_record = {'start': time.time(), 'load': {}, 'process': {}, 'encode': {}, 'decode': {}, 'backward': {}}
        num_views = np.array([len(self.annotations.loc[i]) for i in indices], dtype='uint8')
        return {
            'dataset_name': self.get_dataset_name(),
            'experiment_name': self.get_name(),
            'indices': indices,
            'sub_indices': [self.annotations.loc[i].index.get_level_values(0).to_numpy() for i in indices],
            'num_views': num_views,
            'time': time_record,
            'results': defaultdict(dict),
            'loss': defaultdict(dict),
        }

    def get_batch(self, indices):
        batch = self.get_batch_defaults(indices=indices)

        # It is very important to first load the images, then load other modalities

        batch_data = [modality.get_batch(batch) for modality_name, modality in self.explicit_modalities.items()]

        try:
            for data in batch_data:
                batch.update(data)
        except Exception as e:
            raise RuntimeError(f'Failed to retrieve batch for {self.get_dataset_name()}: {e}')

        return batch

    def setup_annotations(self):
        rel_path = self.get_cfgs('annotations_path')
        fm = File_Manager()
        self.annotations = fm.read_csv_annotations(
            dataset_name=self.dataset_name,
            annotations_rel_path=rel_path,
            multi_view_per_sample=self.multi_view_per_sample,
        )

        if self.annotations is None:
            annotations_url = self.get_cfgs('annotations_url')
            available_csvs_str = '\', \''.join(fm.get_available_csvs(self.dataset_name))
            Console_UI().inform_user(
                '"%s" does not exist among the available datasets: \'%s\'.\nDownloading from:\n %s' %
                (rel_path, available_csvs_str, annotations_url))
            fm.download_zip_file(
                url=annotations_url,
                dataset_name=self.get_dataset_name(),
            )

            self.annotations = fm.read_csv_annotations(
                dataset_name=self.dataset_name,
                annotations_rel_path=rel_path,
                multi_view_per_sample=self.multi_view_per_sample,
            )

        if self.get_cfgs('test_run'):
            self.annotations = self.annotations[self.annotations.index.get_level_values(0) < 100]

    def get_cfgs(self, name, default=None):
        if name in self.experiment_cfgs:
            return self.experiment_cfgs[name]
        if name in self.dataset_cfgs:
            return self.dataset_cfgs[name]
        return Global_Cfgs().get(name, default)

    def end_epoch(self, summary: dict, scene_name: str):
        summary['dataset_name'] = self.dataset_name
        summary['experiment_name'] = self.experiment_name
        summary['modalities'] = defaultdict(dict)
        raw_csv_data = []
        for _, modality in self.modalities.items():
            raw_csv_data.extend(modality.get_runtime_values())
            modality.report_epoch_summary(summary)
            if modality.is_explicit_modality() and modality.is_csv():
                if isinstance(modality.content, pd.Series):
                    raw_csv_data.append(modality.content)
                elif isinstance(modality.content, pd.DataFrame):
                    [raw_csv_data.append(modality.content[c]) for c in modality.content]
                else:
                    raise ValueError(f'The content type of {modality.get_name()} is not implemented')

        # We want the column estimates to be close to eachother in the final output
        raw_csv_data.sort(key=lambda v: v.name if hasattr(v, 'name') else -1)

        File_Manager().write_csv_annotation(
            annotations=pd.concat(raw_csv_data, axis=1),
            dataset_name=self.dataset_name,
            experiment_file_name=f'{scene_name}_{self.get_cfgs("annotations_path")}',
        )
        return summary

    def setup_modalities(self):
        try:
            for modality_name, modality_cfgs in self.experiment_cfgs['modalities'].items():
                self.init_modality(modality_name.lower(), modality_cfgs)
        except pd.errors.EmptyDataError as error:
            path = self.get_cfgs('annotations_path')
            ds = self.dataset_name
            msg = f'No data for {ds} in {path} when setting up modality \'{modality_name}\': {error}'
            raise pd.errors.EmptyDataError(msg)

    def init_modality(self, modality_name: str, modality_cfgs: dict = None):
        modality_name = modality_name.lower()
        assert (modality_cfgs is not None), 'modality_cfgs should not be None in %s' % (modality_name)

        start_time = time.time()
        Modality, content, dictionary = get_modality_and_content(
            annotations=self.annotations,
            modality_name=modality_name,
            modality_cfgs=modality_cfgs,
            ignore_index=-100  # The -100 is defined in the loss_cfgs and not available here :-(
        )

        modality = Modality(
            dataset_name=self.dataset_name,
            dataset_cfgs=self.dataset_cfgs,
            experiment_name=self.experiment_name,
            experiment_cfgs=self.experiment_cfgs,
            modality_name=modality_name,
            modality_cfgs=modality_cfgs,
            content=content,
            dictionary=dictionary,
        )

        if modality.is_explicit_modality():
            self.explicit_modalities[modality_name] = modality
            if modality.is_input_modality():
                self.explicit_input_modalities[modality_name] = modality
            elif modality.is_output_modality():
                self.explicit_output_modalities[modality_name] = modality
            else:
                raise BaseException('Explicit Modalities should either be input or output')
        elif modality.is_implicit_modality():
            self.implicit_modalities[modality_name] = modality

        # Add explicit and implicit modalities
        # Todo - Ali: why do we need to have this split? When do we have the case were a modality is neither
        self.modalities.update(self.explicit_modalities)
        self.modalities.update(self.implicit_modalities)

        if not Global_Cfgs().get('silent_init_info'):
            Console_UI().inform_user(
                info='Initializing %s modality in %s in %d milliseconds' % (modality_name, self.get_name(), 1000 *
                                                                            (time.time() - start_time)),
                debug=(modality_cfgs),
            )

    def get_modality(self, modality_name: str, modality_cfgs: dict = None):
        modality_name = modality_name.lower()
        if modality_name not in self.modalities:
            self.init_modality(modality_name, modality_cfgs)

        modality = self.modalities[modality_name]
        if modality_cfgs is not None:
            modality.update_cfgs(modality_cfgs)
        return modality

        return self.modalities[modality_name]

    def get_modalities(self, modality_names):
        modalities = self.get_cfgs('modalities')
        return [self.get_modality(name.lower(), cfgs) for name, cfgs in modalities.items()]

    def get_explicit_classification_modality_names(self):
        return [m.lower() for m in self.get_explicit_modality_names()
                if self.get_modality(m).is_output_modality() and self.get_modality(m).is_classification()]

    def get_explicit_regression_modality_names(self):
        return [m.lower() for m in self.get_explicit_modality_names()
                if self.get_modality(m).is_output_modality() and self.get_modality(m).is_regression()]

    def get_explicit_input_modality_names(self):
        return [m.lower() for m in self.get_explicit_modality_names() if self.get_modality(m).is_input_modality()]

    def get_explicit_modality_names(self):
        return [m.lower() for m, _ in self.experiment_cfgs['modalities'].items()]

    def get_implicit_classification_modality_names(self):
        return [
            self.get_modality(m.lower()).get_implicit_modality_name()
            for m in self.get_explicit_classification_modality_names()
        ]

    def get_implicit_regression_modality_names(self):
        return [
            self.get_modality(m.lower()).get_implicit_modality_name()
            for m in self.get_explicit_regression_modality_names()
        ]

    def get_implicit_input_modality_names(self):
        return [
            self.get_modality(m.lower()).get_implicit_modality_name() for m in self.get_explicit_input_modality_names()
        ]

    def get_implicit_modality_names(self):
        return [self.get_modality(m.lower()).get_implicit_modality_name() for m in self.get_explicit_modality_names()]

    def get_explicit_pseudo_output_modality_names(self):
        return [
            'pseudo_%s' % m.lower()
            for m in self.get_explicit_modality_names()
            if self.get_modality(m).is_output_modality() and self.get_modality(m).has_pseudo_label()
        ]

    def get_implicit_pseudo_output_modality_names(self):
        return ['implicit_%s' % m for m in self.get_explicit_pseudo_output_modality_names()]

    def get_modality_cfgs(self, modality_name):
        modality_name = modality_name.lower()
        # check if modality is already created
        if modality_name.lower() in self.modalities:
            return self.modalities[modality_name.lower()].get_modality_cfgs()
        # check if it's an implicit modality
        # check if modality is pseudo modality
        elif (modality_name.startswith('pseudo_') and modality_name[len('pseudo_'):].lower() in self.modalities):
            modality = self.modalities[modality_name[len('pseudo_'):].lower()]
            original_modality_cfgs = modality.get_modality_cfgs()
            modality_cfgs = {
                'type': 'pseudo_label',
                'consistency': original_modality_cfgs['consistency'],
                'num_channels': modality.get_num_classes(),
                'tensor_shape': list(original_modality_cfgs['tensor_shape']),
            }
            modality = self.get_modality(modality_name, modality_cfgs)
            return modality.get_modality_cfgs()

        for _, modality in self.modalities.items():
            if (modality.is_explicit_modality()
                    and modality.get_implicit_modality_name().lower() == modality_name.lower()):
                return modality.get_implicit_modality_cfgs()
            # check if it's an implicit pseudo modality

        raise BaseException(('Unknown modality "%s" in dataset "%s",' + ' experiment set "%s"') %
                            (modality_name, self.get_dataset_name(), self.get_name()))

    def get_model_cfgs(self, modality_name):
        modality_name = modality_name.lower()
        if modality_name in self.explicit_modalities:
            modality = self.explicit_modalities[modality_name.lower()]
            if modality.is_explicit_modality():
                return self.explicit_modalities[modality_name.lower()].get_model_cfgs()
        if modality_name.startswith('pseudo_'):
            modality_name = modality_name[len('pseudo_'):]
            modality = self.explicit_modalities[modality_name.lower()]
            if modality.is_explicit_modality():
                modality_cfgs = self.explicit_modalities[modality_name.lower()].get_model_cfgs().copy()
                modality_cfgs['heads'] = ['pseudo_%s' % s for s in modality_cfgs['heads']]
                modality_cfgs['tails'] = ['pseudo_%s' % s for s in modality_cfgs['tails']]
                return modality_cfgs
        return None

    def get_model_name(self, modality_name):
        if modality_name in self.explicit_modalities:
            model_name = self.modalities[modality_name.lower()].get_model_name()
            return model_name
        if modality_name.startswith('pseudo_'):
            modality_name = modality_name[len('pseudo_'):]
            model_name = 'pseudo_%s' % self.modalities[modality_name.lower()].get_model_name()
            return model_name
        raise BaseException('Unsupported modality %s' % modality_name)

    def get_loss_name(self, modality_name):
        return self.modalities[modality_name.lower()].get_loss_name()

    def get_implicit_modality_name(self, explicit_modality_name):
        explicit_modality_name = explicit_modality_name.lower()
        if explicit_modality_name in self.modalities:
            modality = self.modalities[explicit_modality_name]
            if modality.is_explicit_modality():
                return modality.get_implicit_modality_name()
            else:
                modality_name = explicit_modality_name[len('pseudo_'):]
                if modality_name in self.modalities:
                    return 'pseudo_%s' % self.modalities[modality_name].get_implicit_modality_name()

    def get_name(self):
        return self.experiment_name

    def get_dataset_name(self):
        return self.dataset_name
