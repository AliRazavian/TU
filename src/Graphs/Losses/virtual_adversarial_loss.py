import networkx as nx

from .base_loss import Base_Loss


class Virtual_Adversarial_Loss(Base_Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_name = self.modality.get_decoder_name().lower()
        self.output_shape = self.get_cfgs('output_shape')
        print(nx.bfs_edge(self.graph.graph.reverse(copy=True), self.modality.get_name()))
