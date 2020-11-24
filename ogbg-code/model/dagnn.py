import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_scatter import scatter_add
from torch_geometric.nn.glob import *
from torch_geometric.nn.inits import uniform, glorot
from torch_geometric.nn import MessagePassing
from src.constants import *
from src.utils_dag import stack_padded
from typing import Optional
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax


class DAGNN(nn.Module):

    def __init__(self, num_vocab, max_seq_len, emb_dim, hidden_dim, out_dim,
                 num_rels=2, w_edge_attr=True, num_layers=2, bidirectional=False, mapper_bias=True,  # bias only for DVAE simulation
                 agg_x=True, agg=NA_ATTN_H, out_wx=True, out_pool_all=True, out_pool=P_MAX, encoder=None, dropout=0.0,
                 word_vectors=None, emb_dims=[], activation=None, num_class=0, recurr=1):
        super().__init__()
        self.num_class = num_class
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len

        if agg_x and hidden_dim < emb_dim:
            raise ValueError('Hidden dimension too small for input.')

        # configuration
        self.agg_x = agg_x  # use input states of predecessors instead of hidden ones
        self.agg_attn = "attn" in agg
        self.agg_attn_x = "_x" in agg
        self.bidirectional = bidirectional
        self.dirs = [0, 1] if bidirectional else [0]
        self.num_layers = num_layers
        self.out_wx = out_wx
        self.output_all = out_pool_all
        self.recurr = recurr

        # dimensions
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.out_hidden_dim = emb_dim * len(self.dirs) + self.hidden_dim * len(self.dirs) * num_layers if out_wx else self.hidden_dim * len(self.dirs) * num_layers  # USING UNIFY*len(self.dirs)
        # self.out_dim = out_dim  # not needed in OGB

        # initial embedding
        self.encoder = encoder if encoder is not None else init_encoder(word_vectors, emb_dims)

        # aggregate
        # agg_x makes only sense in first NN layer we could afterwards automatically use h? but postponing this...
        # (then add pred_dim term directly when looping over layers below)
        num_rels = num_rels if w_edge_attr else 1
        pred_dim = self.emb_dim if self.agg_x else self.hidden_dim
        attn_dim = self.emb_dim if "_x" in agg else self.hidden_dim
        if "self_attn" in agg:
            # it wouldn't make sense to perform attention based on h when aggregating x... so no hidden_dim needed
            self.node_aggr_0 = nn.ModuleList([
                SelfAttnConv(attn_dim, num_relations=num_rels) for _ in range(num_layers)])
            self.node_aggr_1 = nn.ModuleList([
                SelfAttnConv(attn_dim, num_relations=num_rels, reverse=True) for _ in range(num_layers)])
        elif "attn" in agg:
            op = MultAttnConv if "mattn" in agg else AttnConv
            self.node_aggr_0 = nn.ModuleList([
                op(self.emb_dim if l == 0 else attn_dim, pred_dim, num_relations=num_rels, attn_dim=attn_dim) for l in range(num_layers)])
            self.node_aggr_1 = nn.ModuleList([
                op(self.emb_dim if l == 0 else attn_dim, pred_dim, num_relations=num_rels, attn_dim=attn_dim, reverse=True) for l in range(num_layers)])
        elif agg == NA_GATED_SUM:
            self.node_aggr_0 = nn.ModuleList([
                GatedSumConv(pred_dim, num_rels, mapper_bias=mapper_bias) for _ in range(num_layers)])
            self.node_aggr_1 = nn.ModuleList([
                GatedSumConv(pred_dim, num_rels, mapper_bias=mapper_bias, reverse=True) for _ in range(num_layers)])
        else:
            node_aggr = AggConv(agg, num_rels, pred_dim)
            self.node_aggr_0 = self.node_aggr_1 = nn.ModuleList([node_aggr for _ in range(num_layers)])  # just to have same format

        # RNN
        if recurr:
            for i in self.dirs:
                self.__setattr__("cells_{}".format(i), nn.ModuleList(
                    [nn.GRUCell(emb_dim if l == 0 else self.hidden_dim, self.hidden_dim) for l in range(num_layers)]))
        else:
            for i in self.dirs:
                self.__setattr__("cells_{}".format(i), nn.ModuleList(
                    [nn.Linear((emb_dim if l == 0 else self.hidden_dim)+self.hidden_dim, self.hidden_dim) for l in range(num_layers)]))

        # readout
        if out_pool == P_ATTN:
            d = int(self.out_hidden_dim/2) if self.bidirectional and not self.output_all else self.out_hidden_dim
            self.self_attn_linear_out = torch.nn.Linear(d, 1)
            self._readout = self._out_nodes_self_attn
        else:
            self._readout = getattr(tg.nn, 'global_{}_pool'.format(out_pool))

        # output
        # self.out_norm = nn.LayerNorm(self.out_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # self.out_linear = torch.nn.Linear(self.out_hidden_dim, out_dim)
        # self.activation = init_activation(activation, out_dim)

        # OGB

        if self.num_class > 0:  # classification
            self.graph_pred_linear = torch.nn.Linear(self.out_hidden_dim, self.num_class)
        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            if self.num_vocab == 1:  # regression
                self.graph_pred_linear_list.append(torch.nn.Sequential(
                    torch.nn.Linear(self.out_hidden_dim, self.num_vocab), torch.nn.ReLU()))
            else:
                for i in range(max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(self.out_hidden_dim, self.num_vocab))

    def _out_nodes_self_attn(self, h, batch):
        attn_weights = self.self_attn_linear_out(h)
        attn_weights = F.softmax(attn_weights, dim=-1)
        return global_add_pool(attn_weights * h, batch)

    def _get_output_nodes(self, G, reverse=0):
        if reverse:
            layer0 = G.bi_layer_index[0][0] == 0
            layer0 = G.bi_layer_index[0][1][layer0]
            return layer0
        layer0 = G.bi_layer_index[1][0] == 0
        layer0 = G.bi_layer_index[1][1][layer0]
        return layer0

    def forward(self, G):
        # need to create these here since pyg's batching otherwise messes up the indices
        G.bi_layer_index = torch.stack([
            torch.stack([G._bi_layer_idx0, G._bi_layer_index0], dim=0),
            torch.stack([G._bi_layer_idx1, G._bi_layer_index1], dim=0)
        ], dim=0)
        G.bi_layer_parent_index = stack_padded(
            torch.stack([G._bi_layer_parent_idx0, G._bi_layer_parent_index0], dim=0),
            torch.stack([G._bi_layer_parent_idx1, G._bi_layer_parent_index1], dim=0)
        )

        device = G.x.device
        num_nodes_batch = G.x.shape[0]
        num_layers_batch = max(G.bi_layer_index[0][0]).item() + 1

        G.x = self.encoder(G.x, G.node_depth.view(-1, ))

        G.h = [[torch.zeros(num_nodes_batch, self.hidden_dim).to(device)
                for _ in self.__getattr__("cells_{}".format(0))] for _ in self.dirs]

        for d in self.dirs:
            for l_idx in range(num_layers_batch):
                layer = G.bi_layer_index[d][0] == l_idx
                layer = G.bi_layer_index[d][1][layer]

                inp = G.x[layer]

                if l_idx > 0:  # no predecessors at first layer
                    le_idx = []
                    for n in layer:
                        ne_idx = G.edge_index[1-d] == n
                        le_idx += [ne_idx.nonzero().squeeze(-1)]
                    le_idx = torch.cat(le_idx, dim=-1)
                    lp_edge_index = G.edge_index[:, le_idx]

                    if self.agg_x:
                        # it wouldn't make sense to perform attention based on h when aggregating x... so no h needed
                        kwargs = {"h_attn": G.x, "h_attn_q": G.x} if self.agg_attn else {}  # just ignore query arg if self attn
                        node_agg = self.__getattr__("node_aggr_{}".format(d))[0]
                        ps_h = node_agg(G.x, lp_edge_index, edge_attr=G.edge_attr[le_idx], **kwargs)[layer]
                        # if we aggregate x...
                        s = ps_h.shape
                        if s[-1] < self.hidden_dim:
                            ps_h = torch.cat([ps_h, torch.zeros(s[0], self.hidden_dim-s[1])], dim=-1)
                        # print(G.x[lp_idx])
                        # print(ps_h)

                for i, cell in enumerate(self.__getattr__("cells_{}".format(d))):
                    if l_idx == 0:
                        ps_h = None if self.recurr else torch.zeros(inp.shape[0], self.hidden_dim).to(device)
                    elif not self.agg_x:
                        kwargs = {} if not self.agg_attn else \
                                    {"h_attn": G.x, "h_attn_q": G.x} if self.agg_attn_x else \
                                    {"h_attn": G.h[d][i], "h_attn_q": G.h[d][i-1] if i > 0 else G.x}  # just ignore query arg if self attn
                        node_agg = self.__getattr__("node_aggr_{}".format(d))[i]
                        ps_h = node_agg(G.h[d][i], lp_edge_index, edge_attr=G.edge_attr[le_idx], **kwargs)[layer]

                    inp = cell(inp, ps_h) if self.recurr else cell(torch.cat([inp, ps_h], dim=1))
                    G.h[d][i][layer] += inp

        if self.bidirectional and not self.output_all:
            index = self._get_output_nodes(G)
            h0 = torch.cat([G.x] + [G.h[0][l] for l in range(self.num_layers)], dim=-1) if self.out_wx else \
                torch.cat([G.h[0][l] for l in range(self.num_layers)], dim=-1)
            out0 = self._readout(h0[index], G.batch[index])
            index = self._get_output_nodes(G, reverse=1)
            h1 = torch.cat([G.x] + [G.h[1][l] for l in range(self.num_layers)], dim=-1) if self.out_wx else \
                torch.cat([G.h[1][l] for l in range(self.num_layers)], dim=-1)
            out1 = self._readout(h1[index], G.batch[index])
            out = torch.cat([out0, out1], dim=-1)
        else:
            G.h = torch.cat([G.x] + [G.h[d][l] for d in self.dirs for l in range(self.num_layers)], dim=-1) if self.out_wx else \
                torch.cat([G.h[d][l] for d in self.dirs for l in range(self.num_layers)], dim=-1) if self.bidirectional else \
                    torch.cat([G.h[0][l] for l in range(self.num_layers)], dim=-1)

            if not self.output_all:
                index = self._get_output_nodes(G)
                G.h, G.batch = G.h[index], G.batch[index]
            out = self._readout(G.h, G.batch)

        # out = self.out_linear(out)  #self.out_norm(out)
        out = self.dropout(out)
        # return self.activation(out).squeeze(-1)
        # return out

        if self.num_class > 0:
            return self.graph_pred_linear(out)

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](out))
        return pred_list


def init_encoder(word_vectors, emb_dim):
    if word_vectors is not None:
        return nn.EmbeddingBag.from_pretrained(word_vectors, freeze=True, mode="sum")
    elif len(emb_dim) > 0:
        return nn.EmbeddingBag(emb_dim[0], emb_dim[1], mode="sum")
    return None

def init_param_emb(size, device):
    param = torch.zeros(size).to(device)
    glorot(param)
    # uniform(size, param)
    return param


class AggConv(MessagePassing):
    def __init__(self, agg, num_relations=1, emb_dim=0, reverse=False):
        super(AggConv, self).__init__(aggr=agg, flow='target_to_source' if reverse else 'source_to_target')

        if num_relations > 1:
            assert emb_dim > 0
            self.edge_encoder = torch.nn.Linear(num_relations, emb_dim)  # assuming num_relations one hot encoded
            self.wea = True
        else:
            self.wea = False

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr if self.wea else x_j

    def update(self, aggr_out):
        return aggr_out


class GatedSumConv(MessagePassing):  # dvae needs outdim parameter
    def __init__(self, emb_dim, num_relations=1, mapper_bias=True, reverse=False):
        super(GatedSumConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert emb_dim > 0
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, emb_dim)
        else:
            self.wea = False
        self.mapper = nn.Linear(emb_dim, emb_dim, bias=mapper_bias)
        self.gate = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        h_j = x_j + edge_attr if self.wea else x_j
        return self.gate(h_j) * self.mapper(h_j)

    def update(self, aggr_out):
        return aggr_out


class SelfAttnConv(MessagePassing):
    def __init__(self, emb_dim, attn_dim=0, num_relations=1, reverse=False):
        super(SelfAttnConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert emb_dim > 0
        attn_dim = attn_dim if attn_dim > 0 else emb_dim
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, attn_dim)
        else:
            self.wea = False
        self.attn_lin = nn.Linear(attn_dim, 1)

    # h_attn, edge_attr are optional
    def forward(self, h, edge_index, edge_attr=None, h_attn=None, **kwargs):
        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, h=h, edge_attr=edge_embedding, h_attn=h_attn)

    def message(self, h_j, edge_attr, h_attn_j, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        h_attn = h_attn_j if h_attn_j is not None else h_j
        h_attn = h_attn + edge_attr if self.wea else h_attn
        # have to do part of this here instead of pre-computing a in forward because of missing edges in forward
        # in our dags there is not much overlap in one convolution step, so not much overhead here
        # and if attn transformation linear is applied in forward we'd have to consider full X/H matrices
        # which in our case can be a lot larger
        # BUT we could move it to forward similar to pyg GAT implementation
        # ie apply two different linear to each respectively X/H, edge_attrs which yield a scalar each
        # the in message only sum those up (to obtain a single scalar) and do softmax
        a_j = self.attn_lin(h_attn)
        a_j = softmax(a_j, index, ptr, size_i)
        t = h_j * a_j
        return t

    def update(self, aggr_out):
        return aggr_out


#  simpler version where attn always based on vectors that are also aggregated
# class SelfAttnConv(MessagePassing):
#     def __init__(self, emb_dim, num_relations=1, reverse=False):
#         super(SelfAttnConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')
#
#         assert emb_dim > 0
#         self.edge_encoder = torch.nn.Linear(num_relations, emb_dim) if num_relations > 1 else None
#         self.attn_lin = nn.Linear(emb_dim, 1)
#
#     def forward(self, x, edge_index, edge_attr=None, **kwargs):
#         edge_embedding = self.edge_encoder(edge_attr) if self.edge_encoder is not None else None
#         return self.propagate(edge_index, x=x, edge_attr=edge_embedding)
#
#     def message(self, x_j, edge_attr, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
#         h_j = x_j + edge_attr if edge_attr is not None else x_j
#         # have to to this here instead of pre-computing a in forward because of missing edges in forward
#         # we could do it in forward, but in our dags there is not much overlap in one convolution step
#         # and if attn transformation linear is applied in forward we'd have to consider full X/H matrices
#         # which in our case can be a lot larger
#         # BUT we could move it to forward similar to pyg GAT implementation
#         # ie apply two different linear to each respectively X/H, edge_attrs which yield a scalar each
#         # the in message only sum those up (to obtain a single scalar) and do softmax
#         a_j = self.attn_lin(h_j)
#         a_j = softmax(a_j, index, ptr, size_i)
#         t = x_j * a_j
#         return t
#
#     def update(self, aggr_out):
#         return aggr_out


class AttnConv(MessagePassing):
    def __init__(self, attn_q_dim, emb_dim, attn_dim=0, num_relations=1, reverse=False):
        super(AttnConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert attn_q_dim > 0  # for us is not necessarily equal to attn dim at first RN layer
        assert emb_dim > 0
        attn_dim = attn_dim if attn_dim > 0 else emb_dim
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, attn_dim)
        else:
            self.wea = False
        self.attn_lin = nn.Linear(attn_q_dim + attn_dim, 1)

    # h_attn_q is needed; h_attn, edge_attr are optional (we just use kwargs to be able to switch node aggregator above)
    def forward(self, h, edge_index, h_attn_q=None, edge_attr=None, h_attn=None, **kwargs):
        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, h_attn_q=h_attn_q, h=h, edge_attr=edge_embedding, h_attn=h_attn)

    def message(self, h_attn_q_i, h_j, edge_attr, h_attn_j, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        h_attn = h_attn_j if h_attn_j is not None else h_j
        h_attn = h_attn + edge_attr if self.wea else h_attn
        # see comment in above self attention why this is done here and not in forward
        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        a_j = softmax(a_j, index, ptr, size_i)
        t = h_j * a_j
        return t

    def update(self, aggr_out):
        return aggr_out


class MultAttnConv(MessagePassing):
    def __init__(self, attn_q_dim, emb_dim, attn_dim=0, num_relations=1, reverse=False):
        super(MultAttnConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert attn_q_dim > 0  # for us is not necessarily equal to attn dim at first RN layer
        assert emb_dim > 0
        attn_dim = attn_dim if attn_dim > 0 else emb_dim
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, attn_dim)
        else:
            self.wea = False
        self.attn_linl = nn.Linear(attn_q_dim, attn_q_dim)
        self.attn_linr = nn.Linear(attn_dim, attn_q_dim)

    # h_attn_q is needed; h_attn, edge_attr are optional (we just use kwargs to be able to switch node aggregator above)
    def forward(self, h, edge_index, h_attn_q=None, edge_attr=None, h_attn=None, **kwargs):
        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, h_attn_q=h_attn_q, h=h, edge_attr=edge_embedding, h_attn=h_attn)

    def message(self, h_attn_q_i, h_j, edge_attr, h_attn_j, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        h_attn = h_attn_j if h_attn_j is not None else h_j
        h_attn = h_attn + edge_attr if self.wea else h_attn
        # see comment in above self attention why this is done here and not in forward
        a_j = torch.sum(self.attn_linl(h_attn_q_i) * self.attn_linr(h_attn), dim=1).unsqueeze(-1)
        a_j = softmax(a_j, index, ptr, size_i)
        t = h_j * a_j
        return t

    def update(self, aggr_out):
        return aggr_out


