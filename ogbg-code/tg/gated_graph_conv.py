from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.nn.inits import uniform

SMALL_NUMBER = 1e-7

class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.

    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, out_channels: int, num_layers: int, num_rels = 1, residual_channels = 0, aggr: str = 'add',
                 bias: bool = True, weight_dropout = 0.2, **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_rels = num_rels

        self.weight = Param(Tensor(num_layers, num_rels, out_channels, out_channels))
        # self.weight = \
        F.dropout(self.weight, p=weight_dropout, training=True) #self.training)

        self.rnn = torch.nn.GRUCell(out_channels+residual_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr = None, residual_states = None) -> Tensor:
        """"""
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        incoming_messages = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(self.num_layers):
            for j in range(self.num_rels):
                eidx = edge_index[:, edge_attr[:, j].bool()] if edge_attr is not None else edge_index  # in latter case assume 1 rel
                incoming_messages += self.propagate(eidx, x=x, edge_weight=self.weight[i][j])

            ones = torch.ones(edge_index.shape[1], dtype=x.dtype, device=x.device)
            num_incoming_edges = scatter_add(ones, edge_index[1])
            incoming_messages /= num_incoming_edges.unsqueeze(-1) + SMALL_NUMBER

            incoming_information = torch.cat(residual_states + [incoming_messages], axis=-1)
            x = self.rnn(incoming_information, x)

        return x

    # we do the multiplication here because we assume we have a lot more nodes than messages
    def message(self, x_j: Tensor, edge_weight: OptTensor):
        x_j = torch.matmul(x_j, edge_weight)  # TODO test
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)
