import numpy as np
import torch.nn as nn

from .Networks.neural_net_set import Neural_Net_Set
from .Networks.neural_net import Neural_Net

from .Blocks.basic_conv_block import Basic_Conv_Block
from .Blocks.fully_connected import Fully_Connected
from .base_network_factory import Base_Network_Factory


class Morph_Factory(Base_Network_Factory):

    def get_neural_net(
            self,
            heads: dict,
            tails: dict,
            model_cfgs: dict,
            optimizer_type: str,
            neural_net_type='encoder',
    ):
        assert (len(tails) == 1)
        head_names = list(heads.keys())
        tail_name = list(tails.keys())[0]
        heads = list(heads.values())
        tail = list(tails.values())[0]

        input_shapes = [head['tensor_shape'] for head in heads]
        output_shape = tail['tensor_shape']
        input_consistencies = [head['consistency'] for head in heads]
        output_consistency = tail['consistency']
        are_heads_input_modality = [
            'modality' in head and head['modality'].lower() == 'input'.lower() for head in heads
        ]

        neural_net_names = self.get_neural_net_names(input_shapes=input_shapes,
                                                     output_shape=output_shape,
                                                     input_names=head_names,
                                                     output_name=tail_name,
                                                     neural_net_type=neural_net_type)

        for i in range(len(neural_net_names)):
            neural_net_name = neural_net_names[i]

            if neural_net_name not in self._neural_nets:
                if neural_net_type.lower() == 'encoder':
                    self._neural_nets[neural_net_name] = \
                        self.__init_neural_net(input_shape=input_shapes[i],
                                               output_shape=output_shape,
                                               input_consistency=input_consistencies[i],
                                               output_consistency=output_consistency,
                                               input_name=heads[i]['encoder_name'],
                                               output_name=tail['encoder_name'],
                                               neural_net_name=neural_net_name,
                                               optimizer_type=optimizer_type,
                                               is_head_input_modality=are_heads_input_modality[i])

                if neural_net_type.lower() == 'decoder':
                    self._neural_nets[neural_net_name] = \
                        self.__init_neural_net(input_shape=output_shape,
                                               output_shape=input_shapes[i],
                                               input_consistency=output_consistency,
                                               output_consistency=input_consistencies[i],
                                               input_name=tail['decoder_name'],
                                               output_name=heads[i]['decoder_name'],
                                               neural_net_name=neural_net_name,
                                               optimizer_type=optimizer_type,
                                               is_head_input_modality=False)

        return Neural_Net_Set([self._neural_nets[neural_net_name] for neural_net_name in neural_net_names])

    def __init_neural_net(
            self,
            input_shape,
            output_shape,
            input_consistency,
            output_consistency,
            input_name,
            output_name,
            optimizer_type,
            neural_net_name,
            is_head_input_modality,
    ):
        neural_net_cfgs = {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'add_relu': not is_head_input_modality,
            'neural_net_type': 'morph'
        }

        if (input_shape == output_shape):
            layers = nn.Dropout(p=0, inplace=True)  # Just identity mapping
        elif (input_consistency == output_consistency):
            if (input_shape[0] == output_shape[0]):
                layers = nn.Dropout(p=0, inplace=True)  # Just identity mapping
            else:
                layers = Basic_Conv_Block(input_c=input_shape[0],
                                          output_c=output_shape[0],
                                          kernel_size=1,
                                          consistency=input_consistency,
                                          add_relu=not is_head_input_modality)

            # fix pooling size
        else:
            layers = Fully_Connected(input_shape=input_shape, output_shape=output_shape, num_hidden_layers=0)

        return Neural_Net(neural_net_name=neural_net_name,
                          neural_net_cfgs=neural_net_cfgs,
                          input_name=input_name,
                          output_name=output_name,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          layers=layers,
                          optimizer_type=optimizer_type)

    def get_decoder(
            self,
            input_c,
            output_c,
            input_name,
            output_name,
            neural_net_name,
            optimizer_type,
            is_input_modality=False,
    ):
        pass

    def get_neural_net_names(
            self,
            input_shapes,
            output_shape,
            input_names,
            output_name,
            neural_net_type,
    ):
        names = []
        for input_name, input_shape in zip(input_names, input_shapes):
            names.append('morph_%s_%s_%s_%s_%s' %
                         (input_name.lower(), 'x'.join(str(x) for x in input_shape), output_name.lower(), 'x'.join(
                             str(x) for x in output_shape), neural_net_type.lower()))
        return names

    def update_modality_dims(
            self,
            neural_net_cfgs: dict,
            heads: list,
            tails: list,
            graph,
    ):
        tail = tails[0]
        tail_consistency = graph.nodes[tail]['consistency']
        tail_shape = None
        for head in heads:
            if graph.nodes[head]['consistency'] == tail_consistency:
                if tail_shape is None:
                    tail_shape = np.array(graph.nodes[head]['tensor_shape'].copy())
                else:
                    tail_shape = np.max([tail_shape, np.array(graph.nodes[head]['tensor_shape'].copy())], axis=0)

        graph.nodes[tail]['tensor_shape'] = list(tail_shape)
