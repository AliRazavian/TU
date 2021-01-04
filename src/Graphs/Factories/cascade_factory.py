from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .Networks.neural_net import Neural_Net
from .Blocks.basic_conv_block import Basic_Conv_Block
from .Blocks.resnet_basic_block import ResNet_Basic_Block
from .base_network_factory import Base_Network_Factory


class Cascade_Factory(Base_Network_Factory):

    def get_neural_net(
            self,
            heads: dict,
            tails: dict,
            model_cfgs: dict,
            optimizer_type: str,
            neural_net_type='encoder',
    ):
        assert (len(heads) == 1)
        assert (len(tails) == 1)
        head_name = list(heads.keys())[0]
        tail_name = list(tails.keys())[0]
        head = list(heads.values())[0]
        tail = list(tails.values())[0]
        head_shape = head['tensor_shape']
        tail_shape = tail['tensor_shape']
        input_c = head_shape[0]

        neural_net_cfgs = model_cfgs['neural_net_cfgs']

        block_counts = neural_net_cfgs['block_counts']
        block_output_cs = neural_net_cfgs['block_output_cs']
        if 'kernel_sizes' not in neural_net_cfgs:
            neural_net_cfgs['kernel_sizes'] = [3] * len(block_counts)
        kernel_sizes = neural_net_cfgs['kernel_sizes']
        block_type = neural_net_cfgs['block_type']
        consistency = head['consistency']
        is_head_input_modality = 'modality' in head and \
            head['modality'].lower() == 'input'.lower()

        block_infos = zip(block_counts, block_output_cs, kernel_sizes)

        neural_net_name = self.get_neural_net_name(input_c=input_c,
                                                   input_name=head_name,
                                                   output_name=tail_name,
                                                   consistency=consistency,
                                                   block_type=block_type,
                                                   block_infos=block_infos,
                                                   neural_net_type=neural_net_type)

        if neural_net_name not in self._neural_nets:
            if neural_net_type.lower() == 'encoder':
                self._neural_nets[neural_net_name] = \
                    self.__get_encoder(input_c=input_c,
                                       consistency=consistency,
                                       input_name=head['encoder_name'],
                                       output_name=tail['encoder_name'],
                                       input_shape=head_shape,
                                       output_shape=tail_shape,
                                       neural_net_name=neural_net_name,
                                       neural_net_cfgs=neural_net_cfgs,
                                       optimizer_type=optimizer_type,
                                       is_head_input_modality=is_head_input_modality)

            if neural_net_type.lower() == 'decoder':
                self._neural_nets[neural_net_name] = \
                    self.__get_decoder(output_c=input_c,
                                       consistency=consistency,
                                       input_name=tail['decoder_name'],
                                       output_name=head['decoder_name'],
                                       input_shape=tail_shape,
                                       output_shape=head_shape,
                                       neural_net_name=neural_net_name,
                                       neural_net_cfgs=neural_net_cfgs,
                                       optimizer_type=optimizer_type)

        return self._neural_nets[neural_net_name]

    def __get_encoder(
            self,
            input_c,
            consistency,
            input_name,
            output_name,
            input_shape,
            output_shape,
            neural_net_name,
            neural_net_cfgs,
            optimizer_type,
            is_head_input_modality,
    ):
        kernel_sizes = neural_net_cfgs['kernel_sizes']
        block_counts = neural_net_cfgs['block_counts']
        block_output_cs = neural_net_cfgs['block_output_cs']
        add_max_pool_after_each_block = 'add_max_pool_after_each_block' not in neural_net_cfgs or\
                                        neural_net_cfgs['add_max_pool_after_each_block']

        output_shapes = self.get_output_shapes(shape=input_shape,
                                               num_blocks=len(block_output_cs),
                                               add_max_pool_after_each_block=add_max_pool_after_each_block)[1:]
        neural_net_cfgs['output_shapes'] = output_shapes
        neural_net_cfgs['is_head_input_modality'] = is_head_input_modality
        neural_net_cfgs['input_c'] = input_c
        block_infos = zip(range(len(block_counts)), block_counts, block_output_cs, kernel_sizes, output_shapes)
        block_type = neural_net_cfgs['block_type']

        if consistency.lower() == "1D".lower():
            self.MaxPool = nn.AdaptiveMaxPool1d
        elif consistency.lower() == "2D".lower():
            self.MaxPool = nn.AdaptiveMaxPool2d
        elif consistency.lower() == "3D".lower():
            self.MaxPool = nn.AdaptiveMaxPool3d
        else:
            raise BaseException('Unknown consistency :%s' % consistency)

        if block_type.lower() == 'Basic'.lower():
            Block = Basic_Conv_Block
        elif block_type.lower() == 'ResNetBasic'.lower():
            Block = ResNet_Basic_Block
        else:
            raise BaseException('Unknown block type %s' % (block_type))

        add_relu = not is_head_input_modality

        layers = OrderedDict({})
        for i, block_count, output_c, kernel_size, output_sh in block_infos:
            for j in range(block_count - 1):
                layers['l%d_i%d_j%d_o%d_k%d' % (i, j, input_c, output_c, kernel_size)] = \
                    Block(input_c=input_c,
                          output_c=input_c,
                          kernel_size=kernel_size,
                          consistency=consistency,
                          add_relu=add_relu)
                add_relu = True

            layers['l%d_i%d_j%d_o%d_k%d' % (
                i, block_count - 1, input_c, output_c, kernel_size)] = \
                Block(input_c=input_c,
                      output_c=output_c,
                      kernel_size=kernel_size,
                      consistency=consistency,
                      add_relu=add_relu)

            add_relu = True
            input_c = output_c
            layers['dn_%d' % i] = self.MaxPool(output_sh)

        return Neural_Net(neural_net_name=neural_net_name,
                          neural_net_cfgs=neural_net_cfgs,
                          input_name=input_name,
                          output_name=output_name,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          layers=nn.Sequential(layers),
                          optimizer_type=optimizer_type)

    def __get_decoder(
            self,
            output_c,
            consistency,
            input_name,
            output_name,
            input_shape,
            output_shape,
            neural_net_name,
            neural_net_cfgs,
            optimizer_type,
    ):
        input_c = neural_net_cfgs['block_output_cs'][-1]
        kernel_sizes = list(reversed(neural_net_cfgs['kernel_sizes']))
        block_counts = list(reversed(neural_net_cfgs['block_counts']))
        block_output_cs = list(reversed(neural_net_cfgs['block_output_cs'][:-1]))
        block_output_cs.append(output_c)
        add_max_pool_after_each_block = 'add_max_pool_after_each_block' not in neural_net_cfgs or\
                                        neural_net_cfgs['add_max_pool_after_each_block']

        output_shapes = list(
            reversed(
                self.get_output_shapes(shape=output_shape,
                                       num_blocks=len(block_output_cs),
                                       add_max_pool_after_each_block=add_max_pool_after_each_block)))[1:]

        block_infos = zip(range(len(block_counts)), block_counts, block_output_cs, kernel_sizes, output_shapes)

        block_type = neural_net_cfgs['block_type']

        if block_type.lower() == 'Basic'.lower():
            Block = Basic_Conv_Block
        elif block_type.lower() == 'ResNetBasic'.lower():
            Block = ResNet_Basic_Block
        else:
            raise BaseException('Unknown block type %s' % (block_type))

        layers = OrderedDict({})
        for i, block_count, output_c, kernel_size, output_sh in block_infos:
            layers['dn_%d' % i] = torch.nn.Upsample(output_sh)
            layers['l%d_i%d_j%d_o%d_k%d' % (
                i, 0, input_c, output_c, kernel_size)] = \
                Block(input_c=input_c,
                      output_c=output_c,
                      kernel_size=kernel_size,
                      consistency=consistency,
                      add_relu=True)
            for j in range(block_count - 1):
                layers['l%d_i%d_j%d_o%d_k%d' % (i, j + 1, input_c, output_c, kernel_size)] = \
                    Block(input_c=output_c,
                          output_c=output_c,
                          kernel_size=kernel_size,
                          consistency=consistency,
                          add_relu=True)
            input_c = output_c

        return Neural_Net(neural_net_name=neural_net_name,
                          neural_net_cfgs=neural_net_cfgs,
                          input_name=input_name,
                          output_name=output_name,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          layers=nn.Sequential(layers),
                          optimizer_type=optimizer_type)

    def get_neural_net_name(
            self,
            input_c,
            input_name,
            output_name,
            consistency,
            block_type,
            block_infos,
            neural_net_type,
    ):
        return '%s%s_%d_%s_%s_%s_%s' % (block_type, consistency, input_c, '_'.join(
            '%s-%s-%s' % (str(i), str(j), str(k))
            for i, j, k in block_infos), input_name, neural_net_type.lower(), output_name)

    def update_modality_dims(self, neural_net_cfgs: dict, heads: list, tails: list, graph):
        output_c = neural_net_cfgs['block_output_cs'][-1]
        add_max_pool_after_each_block = 'add_max_pool_after_each_block' not in neural_net_cfgs or\
                                        neural_net_cfgs['add_max_pool_after_each_block']
        output_shape = self.get_output_shapes(shape=graph.nodes[heads[0]]['tensor_shape'],
                                              num_blocks=len(neural_net_cfgs['block_output_cs']),
                                              add_max_pool_after_each_block=add_max_pool_after_each_block)
        graph.nodes[tails[0]]['tensor_shape'] = [output_c, *output_shape[-1]]
        graph.nodes[tails[0]]['consistency'] =\
            graph.nodes[heads[0]]['consistency']

    def get_output_shapes(
            self,
            shape,
            num_blocks,
            add_max_pool_after_each_block,
    ):
        # every block will be followed by a max pooling operation,
        shape = shape[1:]
        all_shapes = [shape]
        if isinstance(shape, (list, tuple)):
            shape = np.array(shape, dtype=int)
        for _ in range(num_blocks):
            if add_max_pool_after_each_block:
                shape = shape // 2
                shape[shape < 1] = 1
            all_shapes.append(list(shape))
        return all_shapes
