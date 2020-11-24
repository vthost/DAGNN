import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import torch_geometric as tg
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.glob import *
from torch_geometric.nn import MessagePassing

from models_pyg import DVAE_BN_PYG
from src.constants import *
from batch import Batch


# this model just uses specific D-VAE methods for decoding if X is used for aggregation etc
# can use other model otherwise..
class DAGNN_BN(DVAE_BN_PYG):

    def __init__(self, emb_dim, hidden_dim, out_dim,
                 max_n, nvt, START_TYPE, END_TYPE, hs, nz, num_layers=2, bidirectional=True,
                 agg=NA_ATTN_H, out_wx=False, out_pool_all=False, out_pool=P_MAX, dropout=0.0,
                 num_nodes=8):  # D-VAE SPECIFIC num_nodes
        super().__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional=bidirectional, num_layers=num_layers, aggx=0)

        self.num_nodes = num_nodes  # D-VAE SPECIFIC

        # configuration
        self.agg = agg
        self.agg_attn = "attn" in agg
        self.agg_attn_x = "_x" in agg
        self.bidirectional = bidirectional
        self.dirs = [0, 1] if bidirectional else [0]
        self.num_layers = num_layers
        self.out_wx = out_wx
        self.output_all = out_pool_all

        # dimensions
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.out_hidden_dim = emb_dim + self.hidden_dim * self.num_layers if out_wx else self.hidden_dim * self.num_layers  # D-VAE SPECIFIC, USING UNIFY, no *len(self.dirs)
        # self.out_dim = out_dim  # not needed in OGB

        # aggregate
        num_rels = 1
        pred_dim = self.hidden_dim
        attn_dim = self.emb_dim if "_x" in agg else self.hidden_dim
        if "self_attn" in agg:
            # it wouldn't make sense to perform attention based on h when aggregating x... so no hidden_dim needed
            self.node_aggr_0 = nn.ModuleList([
                SelfAttnConv(attn_dim, num_relations=num_rels) for _ in range(num_layers)])
            self.node_aggr_1 = nn.ModuleList([
                SelfAttnConv(attn_dim, num_relations=num_rels, reverse=True) for _ in range(num_layers)])
        elif "attn" in agg:
            self.node_aggr_0 = nn.ModuleList([
                AttnConv(self.emb_dim if l == 0 else attn_dim, pred_dim, num_relations=num_rels, attn_dim=attn_dim) for l in range(num_layers)])
            self.node_aggr_1 = nn.ModuleList([
                AttnConv(self.emb_dim if l == 0 else attn_dim, pred_dim, num_relations=num_rels, attn_dim=attn_dim, reverse=True) for l in range(num_layers)])
        elif agg == NA_GATED_SUM:
            # D-VAE SPECIFIC, use super's layers since also used in decoding
            self.node_aggr_0 = nn.ModuleList([
                GatedSumConv(pred_dim, num_rels, mapper=self.mapper_forward[l], gate=self.gate_forward[l]) for l in range(num_layers)])
            self.node_aggr_1 = nn.ModuleList([
                GatedSumConv(pred_dim, num_rels, mapper=self.mapper_backward[l], gate=self.gate_backward[l], reverse=True) for l in range(num_layers)])
        else:
            node_aggr = AggConv(agg, num_rels, pred_dim)
            self.node_aggr_0 = nn.ModuleList([node_aggr for _ in range(num_layers)])  # just to have same format
            node_aggr = AggConv(agg, num_rels, pred_dim, reverse=True)
            self.node_aggr_1 = nn.ModuleList([node_aggr for _ in range(num_layers)])  # just to have same format
        # RNN
        self.__setattr__("cells_{}".format(0), self.grue_forward)
        if self.bidirectional:
            self.__setattr__("cells_{}".format(1), self.grue_backward)

        # readout
        self._readout = self._out_nodes_self_attn if out_pool == P_ATTN else getattr(tg.nn, 'global_{}_pool'.format(out_pool))

        # output
        self.dropout = nn.Dropout(dropout)

        self.out_linear = torch.nn.Linear(self.out_hidden_dim, out_dim) if self.num_layers > 1 else None

    def _out_nodes_self_attn(self, h, batch):
        attn_weights = self.self_attn_linear_out(h)
        attn_weights = F.softmax(attn_weights, dim=-1)
        return global_add_pool(attn_weights * h, batch)

    def _get_output_nodes(self, G):
        if self.bidirectional:
            layer0 = G.bi_layer_index[0][0] == 0
            layer0 = G.bi_layer_index[0][1][layer0]
            return torch.cat([G.h[G.target_index], G.h[layer0]], dim=0), \
                   torch.cat([G.batch[G.target_index], G.batch[layer0]], dim=0)

        return G.h[G.target_index], G.batch[G.target_index]

    def forward(self, G):
        device = self.get_device()
        G = G.to(device)

        num_nodes_batch = G.x.shape[0]
        num_layers_batch = max(G.bi_layer_index[0][0]).item() + 1

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

                for i, cell in enumerate(self.__getattr__("cells_{}".format(d))):

                    if l_idx == 0:
                        ps_h = None
                    else:
                        hs = G.h[d][i]
                        kwargs = {} if not self.agg_attn else \
                                    {"h_attn": G.x, "h_attn_q": G.x} if self.agg_attn_x else \
                                    {"h_attn": hs, "h_attn_q": G.h[d][i-1] if i > 0 else G.x}  # just ignore query arg if self attn
                        node_agg = self.__getattr__("node_aggr_{}".format(d))[i]
                        ps_h = node_agg(hs, lp_edge_index, edge_attr=None, **kwargs)[layer]

                    inp = cell(inp, ps_h)
                    G.h[d][i][layer] += inp

        if not self.output_all:
            # D-VAE SPECIFIC - all have same node number
            if self.bidirectional:
                index = [i for i in range(num_nodes_batch) if i % self.num_nodes == 0]
                index1 = [i + (self.num_nodes - 1) for i in range(num_nodes_batch) if i % self.num_nodes == 0]
                h0 = torch.cat([G.h[0][l][index1] for l in range(self.num_layers)], dim=-1)
                h1 = torch.cat([G.h[1][l][index] for l in range(self.num_layers)], dim=-1)
                G.h = torch.cat([h0, h1], dim=-1)
                G.batch = G.batch[index]
                out = self.hg_unify(G.h)  # now includes layer dim reduction
            else:
                index1 = [i + (self.num_nodes - 1) for i in range(num_nodes_batch) if i % self.num_nodes == 0]
                G.h = torch.cat([G.h[0][l][index1] for l in range(self.num_layers)], dim=-1)
                G.batch = G.batch[index1]
                out = self.out_linear(G.h) if self.num_layers > 1 else G.h
        else:
            G.h = torch.cat([G.x] + [G.h[d][l] for d in self.dirs for l in range(self.num_layers)],
                            dim=-1) if self.out_wx else \
                torch.cat([G.h[d][l] for d in self.dirs for l in range(self.num_layers)],
                          dim=-1) if self.bidirectional else \
                    torch.cat([G.h[0][l] for l in range(self.num_layers)], dim=-1)

            if self.bidirectional:
                G.h = self.hg_unify(G.h)
            elif self.num_layers > 1:
                G.h = self.out_linear(G.h)

            out = self._readout(G.h, G.batch)

        # D-VAE SPECIFIC - return embedding
        return out

    def encode(self, G):
        if type(G) != list:
            G = [G]
        # encode graphs G into latent vectors
        b = Batch.from_data_list(G)
        Hg = self(b)
        mu, logvar = self.fc1(Hg), self.fc2(Hg)
        return mu, logvar

    def _ipropagate_to(self, G, v, propagator, H=None, reverse=False):
        assert not reverse
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        # the difference from original D-VAE is using predecessors' X instead of H
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return

        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]

        v_types = [g.vs[v]['type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        Hv = X
        for l in range(self.num_layers):
            istr = str(l)
            H_name = 'H_forward' + istr  # name of the hidden states attribute
            H_name1 = 'H_forward' + str(l-1)
            # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
            if H is None:
                H_pred1 = None
                H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
                if self.agg_attn_x:
                    H_pred1 = [[self._one_hot(g.vs[x]['type'], self.nvt) for x in g.predecessors(v)] for g in G]

                max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
                if max_n_pred == 0:
                    H = self._get_zero_hidden(len(G))
                else:

                    H_pred = [torch.cat(h_pred +
                                        [self._get_zero_hidden((max_n_pred - len(h_pred)))], 0).unsqueeze(0)
                              for h_pred in H_pred]
                    H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * hs

                    if H_pred1 is not None:
                        H_pred1 = [torch.cat(h_pred +
                                            [self._get_zero_x((max_n_pred - len(h_pred)))], 0).unsqueeze(0)
                                  for h_pred in H_pred1]
                        H_pred1 = torch.cat(H_pred1, 0)  # batch * max_n_pred * hs

                    kwargs = {} if not self.agg_attn else \
                            {"h_attn": H_pred1, "h_attn_q": X} if self.agg_attn_x else \
                            {"h_attn": H_pred,
                             "h_attn_q": torch.cat([g.vs[v][H_name1] for g in G], dim=0) if l > 0 else X}  # just ignore query arg if self attn

                    node_agg = self.__getattr__("node_aggr_{}".format(0))[l]
                    H = node_agg(H_pred, None, **kwargs)

            Hv = propagator[l](Hv, H)
            for i, g in enumerate(G):
                g.vs[v][H_name] = Hv[i:i + 1]
        return Hv


class AggConv(MessagePassing):
    def __init__(self, agg, num_relations=1, emb_dim=0, reverse=False):
        super(AggConv, self).__init__(aggr=agg, flow='target_to_source' if reverse else 'source_to_target')

        if num_relations > 1:
            assert emb_dim > 0
            self.edge_encoder = torch.nn.Linear(num_relations, emb_dim)  # assuming num_relations one hot encoded
            self.wea = True
        else:
            self.wea = False
        self.agg = agg

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        if edge_index is None:
            if self.agg == NA_MAX:
                return torch.max(x, dim=1)[0]
            elif self.agg == NA_SUM:
                return torch.sum(x, dim=1)

        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr if self.wea else x_j

    def update(self, aggr_out):
        return aggr_out


class GatedSumConv(MessagePassing):  # dvae needs outdim parameter
    def __init__(self, emb_dim, num_relations=1, reverse=False, mapper=None, gate=None):
        super(GatedSumConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert emb_dim > 0
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, emb_dim)
        else:
            self.wea = False
        self.mapper = nn.Linear(emb_dim, emb_dim) if mapper is None else mapper
        self.gate = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Sigmoid()) if gate is None else gate

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # HACK assume x contains only message sources
        if edge_index is None:
            h = self.gate(x) * self.mapper(x)
            return torch.sum(h, dim=1)

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
        # HACK assume x contains only message sources
        if edge_index is None:
            h_attn = h_attn if h_attn is not None else h
            attn_weights = self.attn_linear(h_attn).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return torch.mm(attn_weights, h)

        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, h=h, edge_attr=edge_embedding, h_attn=h_attn)

    def message(self, h_j, edge_attr, h_attn_j, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        h_attn = h_attn_j if h_attn_j is not None else h_j
        h_attn = h_attn + edge_attr if self.wea else h_attn
        # have to to this here instead of pre-computing a in forward because of missing edges in forward
        # we could do it in forward, but in our dags there is not much overlap in one convolution step
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
        # HACK assume x contains only message sources
        if edge_index is None:
            query = torch.repeat_interleave(h_attn_q, repeats=h_attn.shape[1], dim=0)
            query = query.view(h_attn.shape[0], h_attn.shape[1], -1)
            h_attn = torch.cat((query, h_attn), -1)
            attn_weights = self.attn_lin(h_attn)
            attn_weights = attn_weights.view(h_attn_q.shape[0], -1)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return torch.einsum('bi,bij->bj', attn_weights, h)

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

