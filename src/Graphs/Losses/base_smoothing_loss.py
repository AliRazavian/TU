import networkx as nx

from .base_loss import Base_Loss


class Base_Smoothing_Loss(Base_Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_name = self.modality.get_decoder_name().lower()
        self.output_shape = self.get_cfgs('output_shape')
        self.input_name = self.get_cfgs('input_name', default='image')

    def get_neural_net(self):
        print(
            nx.shortest_simple_paths(G=self.graph.graph.reverse(copy=True),
                                     source=self.output_name,
                                     target=self.input_name))
        # TODO finish this
