import torch.nn as nn

from .Networks.neural_net_set import Neural_Net_Set
from .Networks.neural_net import Neural_Net

from .Blocks.basic_conv_block import Basic_Conv_Block
from .base_network_factory import Base_Network_Factory


class Fork_Factory(Base_Network_Factory):

    def get_neural_net(
            self,
            heads: dict,
            tails: dict,
            model_cfgs: dict,
            optimizer_type: str,
            neural_net_type='encoder',
    ):

        assert (len(heads) == 1)
        head_name = list(heads.keys())[0]
        tail_names = list(tails.keys())
        head = list(heads.values())[0]
        tails = list(tails.values())

        input_shape = head['tensor_shape']
        output_shapes = [tail['tensor_shape'] for tail in tails]

        input_c = head['num_channels']
        output_cs = [tail['num_channels'] for tail in tails]
        consistency = head['consistency']
        is_head_input_modality = 'modality' in head and \
            head['modality'].lower() == 'input'.lower()

        neural_net_names = self.get_neural_net_names(input_c=input_c,
                                                     output_cs=output_cs,
                                                     input_name=head_name,
                                                     output_names=tail_names,
                                                     neural_net_type=neural_net_type)

        for i in range(len(neural_net_names)):
            neural_net_name = neural_net_names[i]

            if neural_net_name not in self._neural_nets:
                if neural_net_type.lower() == 'encoder':
                    self._neural_nets[neural_net_name] = \
                        self.__init_neural_net(input_c=input_c,
                                               output_c=output_cs[i],
                                               consistency=consistency,
                                               input_name=head['encoder_name'],
                                               output_name=tails[i]['encoder_name'],
                                               input_shape=input_shape,
                                               output_shape=output_shapes[i],
                                               neural_net_name=neural_net_name,
                                               optimizer_type=optimizer_type,
                                               is_head_input_modality=is_head_input_modality)

                if neural_net_type.lower() == 'decoder':
                    self._neural_nets[neural_net_name] = \
                        self.__init_neural_net(input_c=output_cs[i],
                                               output_c=input_c,
                                               consistency=consistency,
                                               input_name=tails[i]['decoder_name'],
                                               output_name=head['decoder_name'],
                                               input_shape=output_shapes[i],
                                               output_shape=input_shape,
                                               neural_net_name=neural_net_name,
                                               optimizer_type=optimizer_type,
                                               is_head_input_modality=False)

        return Neural_Net_Set([self._neural_nets[neural_net_name] for neural_net_name in neural_net_names])

    def __init_neural_net(
            self,
            input_c,
            output_c,
            consistency,
            input_name,
            output_name,
            input_shape,
            output_shape,
            optimizer_type,
            neural_net_name,
            is_head_input_modality,
    ):
        if (input_c == output_c):
            layers = nn.Dropout(p=0, inplace=True)  # Just identity mapping
        else:
            layers = Basic_Conv_Block(input_c=input_c,
                                      output_c=output_c,
                                      kernel_size=1,
                                      consistency=consistency,
                                      add_relu=not is_head_input_modality)
        neural_net_cfgs = {
            'neural_net_type': 'fork',
            'input_c': input_c,
            'output_c': output_c,
            'add_relu': not is_head_input_modality
        }
        return Neural_Net(neural_net_name=neural_net_name,
                          neural_net_cfgs=neural_net_cfgs,
                          input_name=input_name,
                          output_name=output_name,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          layers=layers,
                          optimizer_type=optimizer_type)

    def get_neural_net_names(
            self,
            input_c,
            output_cs,
            input_name,
            output_names,
            neural_net_type,
    ):
        names = []
        for output_name, output_c in zip(output_names, output_cs):
            names.append('fork_%s_%d_%s_%d_%s' %
                         (input_name.lower(), input_c, output_name.lower(), output_c, neural_net_type.lower()))
        return names

    def update_modality_dims(
            self,
            neural_net_cfgs: dict,
            heads: list,
            tails: list,
            graph,
    ):
        tensor_shape = graph.nodes[heads[0].lower()]['tensor_shape'].copy()

        for i in range(len(tails)):
            if 'num_channels' in graph.nodes[tails[i].lower()]:
                tensor_shape[0] = graph.nodes[tails[i].lower()]['num_channels']
            graph.nodes[tails[i].lower()]['tensor_shape'] = tensor_shape.copy()
            graph.nodes[tails[i].lower()]['consistency'] =\
                graph.nodes[heads[0].lower()]['consistency']
