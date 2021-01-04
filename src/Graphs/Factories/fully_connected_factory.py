from .Networks.neural_net import Neural_Net
from .Blocks.fully_connected import Fully_Connected
from .base_network_factory import Base_Network_Factory


class Fully_Connected_Factory(Base_Network_Factory):

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
        input_shape = head['tensor_shape']
        output_shape = tail['tensor_shape']

        neural_net_cfgs = model_cfgs['neural_net_cfgs']
        num_hiddens = neural_net_cfgs['num_hidden']

        is_head_input_modality = 'modality' in head and \
            head['modality'].lower() == 'input'.lower()

        neural_net_name = self.get_neural_net_name(input_shape=input_shape,
                                                   input_name=head_name,
                                                   output_shape=output_shape,
                                                   output_name=tail_name,
                                                   neural_net_type=neural_net_type)

        if neural_net_name not in self._neural_nets:
            if neural_net_type.lower() == 'encoder':
                self._neural_nets[neural_net_name] = \
                    self.__init_neural_net(input_shape=input_shape,
                                           output_shape=output_shape,
                                           input_name=head['encoder_name'],
                                           output_name=tail['encoder_name'],
                                           num_hiddens=num_hiddens,
                                           neural_net_name=neural_net_name,
                                           optimizer_type=optimizer_type,
                                           is_head_input_modality=is_head_input_modality)

            if neural_net_type.lower() == 'decoder':
                self._neural_nets[neural_net_name] = \
                    self.__init_neural_net(input_shape=output_shape,  # it's the opposite
                                           output_shape=input_shape,  # it's the opposite
                                           input_name=tail['decoder_name'],
                                           output_name=head['decoder_name'],
                                           num_hiddens=num_hiddens,
                                           neural_net_name=neural_net_name,
                                           optimizer_type=optimizer_type,
                                           is_head_input_modality=False)

        return self._neural_nets[neural_net_name]

    def __init_neural_net(
            self,
            num_hiddens,
            input_name,
            output_name,
            input_shape,
            output_shape,
            neural_net_name,
            optimizer_type,
            is_head_input_modality,
    ):
        neural_net_cfgs = {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'num_hidden_layers': num_hiddens,
            'add_relu': not is_head_input_modality,
            'neural_net_type': 'fully_connected',
        }

        layers = Fully_Connected(input_shape=input_shape,
                                 output_shape=output_shape,
                                 num_hidden_layers=num_hiddens,
                                 add_relu=not is_head_input_modality)

        return Neural_Net(neural_net_name=neural_net_name,
                          neural_net_cfgs=neural_net_cfgs,
                          input_name=input_name,
                          output_name=output_name,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          layers=layers,
                          optimizer_type=optimizer_type)

    def get_neural_net_name(
            self,
            input_shape,
            output_shape,
            input_name,
            output_name,
            neural_net_type,
    ):
        return 'fc_%s_%s_%s_%s_%s' % (input_name, 'x'.join(str(i) for i in input_shape), output_name, 'x'.join(
            str(o) for o in output_shape), neural_net_type.lower())

    def update_modality_dims(
            self,
            neural_net_cfgs: dict,
            heads: list,
            tails: list,
            graph,
    ):
        return
