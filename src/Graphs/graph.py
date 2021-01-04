import os
import traceback
import pandas as pd
import time
import networkx as nx

from global_cfgs import Global_Cfgs
from file_manager import File_Manager
from UIs.console_UI import Console_UI
from .base_graph import Base_Graph
from .Models.model import Model


class Graph(Base_Graph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_frame = None
        self.time_frame_counter = 0
        self.travers_order = None
        self.init_explicit_modalities()

        self.graph = self.init_graph()
        self.graph_travers_order = self.get_graph_traverse_order()

        self.models = {}
        self.init_models_and_adjust_sizes()

        self.init_remaining_modalities()

        self.losses = {}
        self.init_losses()

        # Save mermaidjs description of graph to log folder
        if not os.path.exists(Global_Cfgs().log_folder):
            os.makedirs(Global_Cfgs().log_folder, exist_ok=True)
        fn = 'mermaidjs_{ds_name}_{graph_name}_{exp_name}_{scene_name}.txt'\
            .format(ds_name=self.get_cfgs('dataset_name'),
                    graph_name=self.get_name(),
                    exp_name=self.experiment_name,
                    scene_name=self.scene_cfgs['name'])
        mermain_fn = os.path.join(Global_Cfgs().log_folder, fn)
        with open(mermain_fn, 'w') as mermain_file:
            mermain_file.write(self.convert_to_mermaidjs())
        Console_UI().inform_user(f'Wrote mermaidjs config to {mermain_fn}')

        self.exception_counter = 0

    def train(self, batch=None):
        for model_name in self.graph_travers_order:
            self.models[model_name].train()
        for _, loss in self.losses.items():
            loss.train()

        if batch is not None:
            try:
                self.zero_grad()
                self.encode(batch)
                if self.reconstruction:
                    self.decode(batch)
                loss = self.compute_loss(batch)

                start_time = time.time()
                if loss > 0:
                    loss.backward()
                    self.step()

                batch['time']['backward'][self.get_name()] = {'start': start_time, 'end': time.time()}
                self.collect_runtime_stats(batch)
                self.exception_counter = 0
                return True
            except KeyError as e:
                Console_UI().warn_user(f'Could not find {e} in:')
                Console_UI().warn_user(sorted(batch.keys()))
                Console_UI().inform_user("\n\n Traceback: \n")
                traceback.print_exc()
                raise e
            except Exception as e:
                Console_UI().warn_user(f'** Error while training batch in {self.get_name()} **')
                Console_UI().warn_user(
                    f'Indices: {batch["indices"]} and encoder image shape {batch["encoder_image"].shape}')
                Console_UI().warn_user(f'Error message: {e}')
                Console_UI().inform_user("\n\n Traceback: \n")
                traceback.print_exc()

                self.exception_counter += 1
                if self.exception_counter > 5:
                    raise RuntimeError(f'Error during training: {e}')
                return False

    def collect_runtime_stats(self, batch):
        start_time = batch['time'].pop('start')
        time_dict = {(outerKey, innerKey): values for outerKey, innerDict in batch['time'].items()
                     for innerKey, values in innerDict.items()}

        if self.time_frame is None:
            self.time_frame = pd.DataFrame(time_dict).T - start_time
            self.time_frame_counter = 1
        else:
            if self.time_frame_counter < 3:
                # just discard the first few samples, due to establishing connections
                # with GPU, they are not worth counting.
                self.time_frame = pd.DataFrame(time_dict).T - start_time
            else:
                self.time_frame *= self.time_frame_counter
                self.time_frame += pd.DataFrame(time_dict).T - start_time
                self.time_frame /= (self.time_frame_counter + 1)
            self.time_frame_counter += 1

        batch['time']['true_full_time'] = time.time() - start_time
        batch['time_stats'] = self.time_frame

        # Console_UI().debug(self.convert_to_mermaidjs(to_visualize='gantt'))

    def eval(self, batch=None):
        for model_name in self.graph_travers_order:
            self.models[model_name].eval()
        for _, loss in self.losses.items():
            loss.eval()
        if batch is not None:
            self.encode(batch)
            if self.reconstruction:
                self.decode(batch)
                # Only in evaluation phase we visualize the reconstructed image (if exist)
                Console_UI().add_last_reconstructed_input(batch)
            self.compute_loss(batch)

    def encode(self, batch):
        for model_name in self.graph_travers_order:
            try:
                self.models[model_name].encode(batch)
            except RuntimeError as e:
                msg = f'Got error for model {model_name}: {e} for {len(batch["indices"])} indices.' + \
                      f'\nThe encoder image shape {batch["encoder_image"].shape}'
                key = f'encoder_{model_name[:-5]}'
                if key in batch:
                    msg += f' where the shape of the input was {batch[key].shape}'
                else:
                    msg += f' but the input {key} could not be found :-('
                raise RuntimeError(msg)

    def decode(self, batch):
        for model_name in reversed(self.graph_travers_order):
            self.models[model_name].decode(batch)

    def compute_loss(self, batch, accumulated_loss=0):
        for loss in self.losses.values():
            accumulated_loss += loss(batch) * loss.coef
        return accumulated_loss

    def step(self):
        for model_name in self.graph_travers_order:
            self.models[model_name].step()
        for _, loss in self.losses.items():
            loss.step()

    def zero_grad(self):
        for model_name in self.graph_travers_order:
            self.models[model_name].zero_grad()
        for _, loss in self.losses.items():
            loss.zero_grad()

    def save(self, scene_name):
        Console_UI().inform_user('\n*****************************************' +
                                 f'\nSave network and losses for {scene_name}')
        no_networks = 0
        no_losses = 0
        for model_name in self.graph_travers_order:
            self.models[model_name].save(scene_name)
            no_networks += 1
        for _, loss in self.losses.items():
            loss.save(scene_name)
            no_losses += 1

        Console_UI().inform_user(f'Saved {no_networks} networks and {no_losses} losses to disk' +
                                 f' check out: {File_Manager().get_network_dir_path()}' +
                                 '\n*****************************************\n')

    def update_learning_rate(self, learning_rate):
        learning_rate = learning_rate * (self.experiment_set.batch_size / 128)
        for model_name in self.graph_travers_order:
            self.models[model_name].update_learning_rate(learning_rate)
        for loss_name in self.losses.keys():
            self.losses[loss_name].update_learning_rate(learning_rate)

    def update_stochastic_weighted_average_parameters(self):
        has_run_average = False
        for model_name in self.graph_travers_order:
            model_has_run_average = self.models[model_name].update_stochastic_weighted_average_parameters()
            if model_has_run_average:
                has_run_average = True
        for loss_name in self.losses.keys():
            loss_has_run_average = self.losses[loss_name].update_stochastic_weighted_average_parameters()
            if loss_has_run_average:
                has_run_average = True

        return has_run_average

    def prepare_for_batchnorm_update(self):
        for model_name in self.graph_travers_order:
            self.models[model_name].prepare_for_batchnorm_update()
        for loss_name in self.losses.keys():
            self.losses[loss_name].prepare_for_batchnorm_update()

    def update_batchnorm(self, batch):
        self.encode(batch)
        if self.reconstruction:
            self.decode(batch)
        for model_name in self.graph_travers_order:
            self.models[model_name].update_batchnorm(batch)

        for loss_name in self.losses.keys():
            self.losses[loss_name](batch)
            self.losses[loss_name].update_batchnorm(batch)

    def finish_batchnorm_update(self):
        for model_name in self.graph_travers_order:
            self.models[model_name].finish_batchnorm_update()
        for loss_name in self.losses.keys():
            self.losses[loss_name].finish_batchnorm_update()

    def init_graph(self):
        G = nx.DiGraph()
        for modality_name, modality_cfgs in self.get_modalities().items():
            modality_cfgs.update({'node_type': 'modality'})
            G.add_node(modality_name.lower(), **modality_cfgs)

        for model_name, model_cfgs in self.get_models().items():
            model_cfgs.update({'node_type': 'model'})
            G.add_node(model_name.lower(), **model_cfgs)
            for h in model_cfgs['heads']:
                G.add_edge(h.lower(), model_name.lower())
            for t in model_cfgs['tails']:
                G.add_edge(model_name.lower(), t.lower())

        assert(nx.is_directed_acyclic_graph(G)),\
            'Graph for task "%s" is not DAG'
        return G

    def convert_to_mermaidjs(self, to_visualize='graph'):
        chart = 'graph LR\n'
        gantt = 'gantt\n'

        if to_visualize.lower() == 'graph'.lower():
            for (head, tail) in list(self.graph.edges):
                if self.graph.nodes[tail]['node_type'] == 'modality':
                    chart += '\t%s --> |%s|%s((%s))\n' % (head.replace(
                        ' ', '_'), self.experiment_set.get_modality(tail).get_shape_str(), tail.replace(
                            ' ', '_'), tail.replace('_', '<br/>'))

                elif self.graph.nodes[tail]['node_type'] == 'loss':
                    chart += '\tsubgraph %s\n\t\t%s\n\tend\n' % (tail.replace(
                        ' ', '_'), self.graph.nodes[tail]['modality_name'].replace(' ', '_'))

                else:
                    chart += '\t%s((%s)) --> |%s|%s[%s]\n' % (head.replace(' ', '_'), head.replace(
                        ' ', '_'), self.experiment_set.get_modality(head).get_shape_str(), tail.replace(
                            ' ', '_'), tail.replace('_', '<br/>'))
            return chart.replace('style', 'Style')
        else:
            if self.time_frame is not None:
                x = self.time_frame
                x.sort_values('start', inplace=True)

                gantt += 'title A cycle time in ms\n'
                prev_section = ''
                for i in range(len(x)):
                    t = x.iloc[i]
                    section, name = t.name
                    if section != prev_section:
                        gantt += '\tsection %s\n' % section
                        prev_section = section
                    gantt += '\t\t%s: %04d, %04d\n' % (name, 1000 * t['start'], 1000 * t['end'])
        return gantt.replace('style', 'Style')

    def get_graph_traverse_order(self):
        ordered_nodes = list(nx.topological_sort(self.graph))
        try:
            ordered_models = [m for m in ordered_nodes if self.graph.nodes[m]['node_type'] == 'model']
            return ordered_models
        except KeyError as e:
            Console_UI().warn_user("You have probably missed a key with node_type:")
            Console_UI().warn_user([m for m in ordered_nodes if 'node_type' not in self.graph.nodes[m]])
            raise KeyError(f'Key not found: {e}')
        return None

    def init_models_and_adjust_sizes(self):
        for model_name in self.graph_travers_order:
            self.models[model_name] = \
                Model(graph=self.graph,
                      graph_cfgs=self.graph_cfgs,
                      experiment_set=self.experiment_set,
                      model_name=model_name,
                      model_cfgs=self.graph.nodes[model_name],
                      task_cfgs=self.task_cfgs,
                      scene_cfgs=self.scene_cfgs,
                      scenario_cfgs=self.scenario_cfgs)

    def init_explicit_modalities(self):
        for modality_name, modality_cfgs in self.get_modalities().items():
            if not (modality_cfgs['type'].lower() in ['implicit'.lower()]):
                self.experiment_set.get_modality(modality_name, modality_cfgs)

    def init_remaining_modalities(self):
        ordered_nodes = list(nx.topological_sort(self.graph))
        ordered_modalities = [m for m in ordered_nodes if self.graph.nodes[m]['node_type'] == 'modality']
        for m in ordered_modalities:
            # The get_modality inits but there is no need to use the returned modality here
            self.experiment_set.get_modality(m, self.graph.nodes[m])
            self.graph.nodes[m].update(self.experiment_set.get_modality_cfgs(m))

    def init_losses(self):
        for loss_name, loss_cfgs in self.get_losses().items():
            loss_cfgs.update({'node_type': 'loss'})
            self.graph.add_node(loss_name, **loss_cfgs)
            self.graph.add_edge(loss_cfgs['modality_name'], loss_name)

        ordered_nodes = list(nx.topological_sort(self.graph))
        ordered_losses = [l for l in ordered_nodes if self.graph.nodes[l]['node_type'] == 'loss']

        for loss_name in ordered_losses:
            loss_cfgs = self.graph.nodes[loss_name]
            loss_type = loss_cfgs['loss_type']
            if loss_type.lower() == 'cross_entropy'.lower():
                if not self.classification and not self.pi_model:
                    continue
                from .Losses.cross_entropy_loss import Cross_Entropy_Loss as Loss
            elif loss_type.lower() == 'bipolar_margin_loss'.lower():
                if not self.classification and not self.pi_model:
                    continue
                from .Losses.bipolar_margin_loss import Bipolar_Margin_Loss as Loss
            elif loss_type.lower() == 'kldiv_loss'.lower():
                if not self.classification:
                    continue
                raise NotImplementedError

            elif loss_type.lower() == 'hierarchical_bce'.lower():
                if not self.classification:
                    continue
                from .Losses.hierarchical_BCE_loss import Hierarchical_BCE_Loss as Loss

            elif loss_type.lower() == 'l1_laplacian_pyramid_loss'.lower():
                if not self.reconstruction:
                    continue
                from .Losses.L1_laplacian_pyramid_loss import L1_Laplacian_Pyramid_Loss as Loss
            elif loss_type.lower() == 'l2_loss'.lower():
                if not self.reconstruction:
                    continue
                from .Losses.L2_loss import L2_Loss as Loss
            elif loss_type.lower() == 'wGAN_gp'.lower():
                if (not self.real_fake):
                    continue
                from .Losses.wgan_gp_loss import Wasserstein_GAN_GP_Loss as Loss
            elif loss_type.lower() == 'triplet_metric'.lower():
                if not self.classification:
                    continue
                from .Losses.triplet_metric_loss import Triplet_Metric_Loss as Loss
            elif loss_type.lower() == 'mse_loss'.lower():
                if (not self.regression):
                    continue
                from .Losses.mse_loss import MSE_Loss as Loss
            elif loss_type.lower() == 'mse_with_spatial_transform'.lower():
                if (not self.regression):
                    continue
                from .Losses.mse_wsp_loss import MSE_WSP_Loss as Loss

            else:
                raise BaseException('Unknown loss type: %s' % (loss_type))
            self.losses[loss_name] = \
                Loss(experiment_set=self.experiment_set,
                     graph=self,
                     graph_cfgs=self.graph_cfgs,
                     loss_name=loss_name,
                     loss_cfgs=loss_cfgs,
                     task_cfgs=self.task_cfgs,
                     scene_cfgs=self.scene_cfgs,
                     scenario_cfgs=self.scenario_cfgs)

    def get_losses(self):
        return {
            **self.get_classification_loss(),
            **self.get_reconstruction_loss(),
            **self.get_discriminator_loss(),
            **self.get_identification_loss(),
            **self.get_regression_loss()
        }

    def get_classification_loss(self):
        classification_loss = {}
        if self.classification or self.pi_model:
            for modality_name in self.get_experiment_explicit_modalities():
                modality = self.experiment_set.get_modality(modality_name)
                if modality.has_classification_loss():
                    classification_loss[modality.get_classification_loss_name()] = \
                        modality.get_classification_loss_cfgs()
        return classification_loss

    def get_regression_loss(self):
        reg_loss = {}
        if self.regression:
            for modality_name in self.get_experiment_explicit_modalities():
                modality = self.experiment_set.get_modality(modality_name)
                if modality.has_regression_loss():
                    reg_loss[modality.get_regression_loss_name()] = \
                        modality.get_regression_loss_cfgs()
        return reg_loss

    def get_reconstruction_loss(self):
        reconstruction_loss = {}
        if self.reconstruction:
            for modality_name in self.get_modalities():
                modality = self.experiment_set.get_modality(modality_name)
                if modality.has_reconstruction_loss():
                    if modality.is_input_modality() or modality.is_implicit_modality():
                        reconstruction_loss[modality.get_reconstruction_loss_name()] = \
                            modality.get_reconstruction_loss_cfgs()
        return reconstruction_loss

    def get_discriminator_loss(self):
        discriminator_loss = {}
        if self.real_fake:
            for modality_name in self.get_modalities():
                modality = self.experiment_set.get_modality(modality_name)
                if (modality.is_input_modality() and self.reconstruction) or\
                   modality.has_discriminator_loss():
                    discriminator_loss[modality.get_discriminator_loss_name()] = \
                        modality.get_discriminator_loss_cfgs()
        return discriminator_loss

    def get_identification_loss(self):
        identification_loss = {}
        if self.identification:
            for modality_name in self.get_modalities():
                modality = self.experiment_set.get_modality(modality_name)
                if modality.has_identification_loss():
                    identification_loss[modality.get_identification_loss_name()] = \
                        modality.get_identification_loss_cfgs()

        return identification_loss

    def dropModelNetworks(self):
        """
        In order to save space + make sure we always use the last saved model
        when evaluating it is safest to clear the encoders and decoders and
        reload them from the disk.

        TODO: This could be implemented using a Singleton that has all the models in a dict
        """
        for model_name in self.graph_travers_order:
            self.models[model_name].dropNetworks()

    def reloadModelNetworks(self):
        for model_name in self.graph_travers_order:
            self.models[model_name].dropNetworks()
            self.models[model_name].get_encoder()
            if self.reconstruction:
                self.models[model_name].get_decoder()
