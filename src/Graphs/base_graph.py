import json
from abc import ABCMeta

from global_cfgs import Global_Cfgs
from file_manager import File_Manager
from UIs.console_UI import Console_UI


class Base_Graph(metaclass=ABCMeta):

    def __init__(
            self,
            graph_name,
            experiment_set,
            task_cfgs,
            scene_cfgs,
            scenario_cfgs,
    ):
        self.graph_name = graph_name
        self.task_cfgs = task_cfgs
        self.scene_cfgs = scene_cfgs
        self.scenario_cfgs = scenario_cfgs
        self.experiment_set = experiment_set
        self.experiment_name = self.experiment_set.get_name()

        self.graph_cfgs = self.get_graph_cfgs(self.graph_name)
        self.classification = self.get_cfgs('classification', default=False)
        self.reconstruction = self.get_cfgs('reconstruction', default=False)
        self.identification = self.get_cfgs('identification', default=False)
        self.regression = self.get_cfgs('regression', default=False)
        self.pi_model = self.get_cfgs('pi_model', default=False)
        self.real_fake = self.get_cfgs('real_fake', default=False)
        self.optimizer_type = self.get_cfgs('optimizer_type')

        if not Global_Cfgs().get('silent_init_info'):
            UI = Console_UI()
            UI.inform_user(
                info=['explicit experiment modalities',
                      list(self.get_experiment_explicit_modalities().keys())],
                debug=self.get_experiment_explicit_modalities(),
            )
            UI.inform_user(
                info=['implicit experiment modalities',
                      list(self.get_experiment_implicit_modalities().keys())],
                debug=self.get_experiment_implicit_modalities(),
            )
            UI.inform_user(
                info=['explicit graph modalities',
                      list(self.get_graph_specific_explicit_modalities().keys())],
                debug=self.get_graph_specific_explicit_modalities(),
            )
            UI.inform_user(
                info=['implicit graph modalities',
                      list(self.get_graph_specific_implicit_modalities().keys())],
                debug=self.get_graph_specific_implicit_modalities(),
            )
            UI.inform_user(
                info=['explicit models', list(self.get_explicit_models().keys())],
                debug=self.get_explicit_models(),
            )
            UI.inform_user(
                info=['implicit models', list(self.get_implicit_models().keys())],
                debug=self.get_implicit_models(),
            )

    def get_graph_cfgs(self, graph_name):
        graph_cfgs = File_Manager().read_graph_config(graph_name)
        graph_cfgs = self.fix_experiment_modalities(graph_cfgs)
        return graph_cfgs

    def fix_experiment_modalities(self, graph_cfgs):
        experiment_explicit_input_modalities = \
            self.experiment_set.get_explicit_input_modality_names()
        experiment_explicit_classification_modalities = \
            self.experiment_set.get_explicit_classification_modality_names()
        experiment_explicit_regression_modalities = \
            self.experiment_set.get_explicit_regression_modality_names()

        experiment_explicit_modalities = \
            self.experiment_set.get_explicit_modality_names()

        experiment_implicit_input_modalities = \
            self.experiment_set.get_implicit_input_modality_names()
        experiment_implicit_classification_modalities = \
            self.experiment_set.get_implicit_classification_modality_names()
        experiment_implicit_regression_modalities = \
            self.experiment_set.get_implicit_regression_modality_names()

        experiment_implicit_modalities = \
            self.experiment_set.get_implicit_modality_names()

        experiment_explicit_pseudo_output_modalities = \
            self.experiment_set.get_explicit_pseudo_output_modality_names()

        experiment_implicit_pseudo_output_modalities = \
            self.experiment_set.get_implicit_pseudo_output_modality_names()

        graph_cfgs = json.dumps(graph_cfgs)
        graph_cfgs = graph_cfgs.replace('"EXPERIMENT_EXPLICIT_MODALITIES"', str(experiment_explicit_modalities)[1:-1])
        graph_cfgs = graph_cfgs.replace('"EXPERIMENT_EXPLICIT_INPUT_MODALITIES"',
                                        str(experiment_explicit_input_modalities)[1:-1])

        graph_cfgs = graph_cfgs.replace('"EXPERIMENT_EXPLICIT_CLASSIFICATION_MODALITIES"',
                                        str(experiment_explicit_classification_modalities)[1:-1])
        graph_cfgs = graph_cfgs.replace('"EXPERIMENT_EXPLICIT_REGRESSION_MODALITIES"',
                                        str(experiment_explicit_regression_modalities)[1:-1])

        graph_cfgs = graph_cfgs.replace('"EXPERIMENT_IMPLICIT_MODALITIES"', str(experiment_implicit_modalities)[1:-1])
        graph_cfgs = graph_cfgs.replace('"EXPERIMENT_IMPLICIT_INPUT_MODALITIES"',
                                        str(experiment_implicit_input_modalities)[1:-1])
        graph_cfgs = graph_cfgs.replace('"EXPERIMENT_IMPLICIT_CLASSIFICATION_MODALITIES"',
                                        str(experiment_implicit_classification_modalities)[1:-1])
        graph_cfgs = graph_cfgs.replace('"EXPERIMENT_IMPLICIT_REGRESSION_MODALITIES"',
                                        str(experiment_implicit_regression_modalities)[1:-1])

        graph_cfgs = graph_cfgs.replace('"EXPERIMENT_EXPLICIT_PSEUDO_OUTPUT_MODALITIES"',
                                        str(experiment_explicit_pseudo_output_modalities)[1:-1])

        graph_cfgs = graph_cfgs.replace('"EXPERIMENT_IMPLICIT_PSEUDO_OUTPUT_MODALITIES"',
                                        str(experiment_implicit_pseudo_output_modalities)[1:-1])

        graph_cfgs = graph_cfgs.replace('\'', "\"")
        graph_cfgs = json.loads(graph_cfgs)

        return graph_cfgs

    def get_modalities(self):
        return {**self.get_graph_specific_modalities(), **self.get_experiment_modalities()}

    def get_explicit_modalities(self):
        return {**self.get_experiment_explicit_modalities(), **self.get_graph_specific_explicit_modalities()}

    def get_experiment_modalities(self):
        return {**self.get_experiment_explicit_modalities(), **self.get_experiment_implicit_modalities()}

    def get_experiment_explicit_modalities(self):
        modalities_cfgs = {}
        explicit_modality_names = \
            self.graph_cfgs['modalities']['experiment_modalities']
        for modality_name in explicit_modality_names:
            modalities_cfgs[modality_name] =\
                self.experiment_set.get_modality_cfgs(modality_name)
        return modalities_cfgs

    def get_experiment_implicit_modalities(self):
        implicit_modalities_cfgs = {}
        for modality_name, modality_cfgs in \
                self.get_experiment_explicit_modalities().items():
            explicit_modality = \
                self.experiment_set.get_modality(modality_name, modality_cfgs)
            implicit_modality_name = \
                self.experiment_set.get_implicit_modality_name(explicit_modality.get_name())
            implicit_modalities_cfgs[implicit_modality_name] = \
                self.experiment_set.get_modality_cfgs(implicit_modality_name)
        return implicit_modalities_cfgs

    def get_graph_specific_modalities(self):
        return {**self.get_graph_specific_explicit_modalities(), **self.get_graph_specific_implicit_modalities()}

    def get_pseudo_explicit_modalities(self):
        modalities_cfgs = {}
        return modalities_cfgs

    def get_pseudo_implicit_modalities(self):
        modalities_cfgs = {}
        return modalities_cfgs

    def get_graph_specific_explicit_modalities(self):
        modalities = self.graph_cfgs['modalities']['graph_specific_modalities']
        return {name: cfgs for name, cfgs in modalities.items() if (cfgs['type'].lower() != 'Implicit'.lower())}

    def get_graph_specific_implicit_modalities(self):
        modalities = self.graph_cfgs['modalities']['graph_specific_modalities']
        implicit_modalities_cfgs = {
            name: cfgs for name, cfgs in modalities.items() if (cfgs['type'].lower() == 'Implicit'.lower())
        }

        for modality_name, modality_cfgs in\
                self.get_graph_specific_explicit_modalities().items():
            explicit_modality = \
                self.experiment_set.get_modality(modality_name, modality_cfgs)
            implicit_modalities_cfgs[explicit_modality.get_implicit_modality_name()] = \
                explicit_modality.get_implicit_modality_cfgs()
        return implicit_modalities_cfgs

    def get_models(self):
        return {**self.get_explicit_models(), **self.get_implicit_models()}

    def get_implicit_models(self):
        models_cfgs = {}
        explicit_modalities = self.get_explicit_modalities()
        for modality_name, _ in explicit_modalities.items():
            models_cfgs[self.experiment_set.get_model_name(modality_name)] =\
                self.experiment_set.get_model_cfgs(modality_name)
        return models_cfgs

    def get_explicit_models(self):
        for model_name, model_cfgs in self.graph_cfgs['models']['graph_specific_models'].items():
            if 'remove_from_tails' in model_cfgs:
                model_cfgs['remove_from_tails'] =\
                    [m.lower() for m in model_cfgs['remove_from_tails']]
                model_cfgs['tails'] = [
                    t.lower() for t in model_cfgs['tails'] if t.lower() not in model_cfgs['remove_from_tails']
                ]

        return self.graph_cfgs['models']['graph_specific_models']

    def get_cfgs(self, name, default=None):
        try:
            if name in self.graph_cfgs:
                return self.graph_cfgs[name]
            if name in self.task_cfgs:
                return self.task_cfgs[name]
            if name in self.task_cfgs['apply']:
                return self.task_cfgs['apply'][name]
            if name in self.scene_cfgs:
                return self.scene_cfgs[name]
            if name in self.scenario_cfgs:
                return self.scenario_cfgs[name]
        except TypeError as e:
            Console_UI().inform_user(self.graph_cfgs)
            Console_UI().inform_user(self.task_cfgs)
            Console_UI().inform_user(self.scene_cfgs)
            Console_UI().inform_user(self.scenario_cfgs)
            raise TypeError(f'Error during {self.get_name()}: {e}')

        return Global_Cfgs().get(name)

    def get_name(self):
        return self.graph_name
