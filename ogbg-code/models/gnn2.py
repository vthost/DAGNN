import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, SAGPooling, GraphUNet, DenseSAGEConv, dense_diff_pool, GatedGraphConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_sort_pool, global_mean_pool, global_add_pool
from torch_geometric.utils import dropout_adj, to_dense_batch, to_dense_adj
from math import ceil
import numpy as np
from conv import GCNConv as GCNConvOGB
from torch_geometric.nn import GatedGraphConv, GATConv
from src.tg.gated_graph_conv import GatedGraphConv as GatedGraphConv_EType
from torch_scatter import scatter_add


class GGNN_Simple(nn.Module):

    def __init__(self, num_vocab, max_seq_len, node_encoder, emb_dim,
                 #in_dim, hidden_dim, out_dim, num_rels, word_vectors=None,
                 # residual_connections={"2": [0], "4": [0, 2]},  # For layer i, specify list of layers whose output is added as an input
                 layer_timesteps=[5], num_class=0):
        super(GGNN_Simple, self).__init__()
        self.num_class = num_class  # if we do classification
        self.layer_timesteps = layer_timesteps
        # self.residual_connections = residual_connections

        #     'use_edge_bias': False,
        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.node_encoder = node_encoder

        self.convs = []
        for layer_idx, t in enumerate(layer_timesteps):
            self.convs += [GatedGraphConv(emb_dim, t)]
        self.convs = nn.ModuleList(self.convs)

        self.classifier_l = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim),
            nn.Sigmoid()
        )
        self.classifier_r = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim),
            nn.Tanh()
        )

        if self.num_class > 0:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, self.num_class)
        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()

            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_vocab))


    def forward(self, batched_data):
        x, edge_index, node_depth, batch = batched_data.x, batched_data.edge_index,  batched_data.node_depth, batched_data.batch

        x = self.node_encoder(x, node_depth.view(-1, ))
        node_states_per_layer = []  # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer.append(x)

        for layer_idx, num_timesteps in enumerate(self.layer_timesteps):

            # # Extract residual messages, if any:
            # layer_residual_connections = self.residual_connections.get(str(layer_idx))
            # layer_residual_states = [] if layer_residual_connections is None else \
            #     [node_states_per_layer[residual_layer_idx]
            #                              for residual_layer_idx in layer_residual_connections]

            # Record new states for this layer. Initialised to last state, but will be updated below:
            node_states_layer = self.convs[layer_idx](node_states_per_layer[-1], edge_index)
            node_states_per_layer.append(node_states_layer)

        hx = torch.cat([node_states_per_layer[-1], x], dim=-1)
        x = self.classifier_l(hx) * self.classifier_r(hx)
        output = global_add_pool(x, batch=batch)

        if self.num_class > 0:
            return self.graph_pred_linear(output)

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](output))

        return pred_list


class GGNN(nn.Module):

    def __init__(self, num_vocab, max_seq_len, node_encoder, emb_dim, num_rels,
                 #in_dim, hidden_dim, out_dim, num_rels, word_vectors=None,
                 residual_connections={"2": [0], "4": [0, 2]},  # For layer i, specify list of layers whose output is added as an input
                 layer_timesteps=[2, 2, 1, 2, 1],  # number of layers & propagation steps per layer
                 edge_weight_dropout_keep_prob=.8):
        super(GGNN, self).__init__()

        self.layer_timesteps = layer_timesteps
        self.residual_connections = residual_connections

        #     'use_edge_bias': False,
        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.node_encoder = node_encoder

        self.convs = []
        for layer_idx, t in enumerate(layer_timesteps):
            layer_residual_connections = self.residual_connections.get(str(layer_idx))
            layer_residual_dim = 0 if layer_residual_connections is None else len(layer_residual_connections)*emb_dim
            self.convs += [GatedGraphConv_EType(emb_dim, t, num_rels, layer_residual_dim)]
        self.convs = nn.ModuleList(self.convs)

        self.classifier_l = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim),
            nn.Sigmoid()
        )
        self.classifier_r = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim),
            nn.Tanh()
        )

        self.graph_pred_linear_list = torch.nn.ModuleList()

        for i in range(max_seq_len):
            self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_vocab))


    def forward(self, batched_data):
        x, edge_index, node_depth, batch = batched_data.x, batched_data.edge_index,  batched_data.node_depth, batched_data.batch

        x = self.node_encoder(x, node_depth.view(-1, ))
        node_states_per_layer = []  # one entry per layer (final state of that layer), shape: number of nodes in batch v x D
        node_states_per_layer.append(x)

        for layer_idx, num_timesteps in enumerate(self.layer_timesteps):

            # Extract residual messages, if any:
            layer_residual_connections = self.residual_connections.get(str(layer_idx))
            layer_residual_states = [] if layer_residual_connections is None else \
                [node_states_per_layer[residual_layer_idx]
                                         for residual_layer_idx in layer_residual_connections]

            # Record new states for this layer. Initialised to last state, but will be updated below:
            node_states_layer = self.convs[layer_idx](node_states_per_layer[-1], edge_index, batched_data.edge_attr, layer_residual_states)
            node_states_per_layer.append(node_states_layer)

        hx = torch.cat([node_states_per_layer[-1], x], dim=-1)
        x = self.classifier_l(hx) * self.classifier_r(hx)
        output = global_add_pool(x, batch=batch)

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](output))

        return pred_list


# https://arxiv.org/pdf/1710.10903.pdf
# For the inductive learning task, we apply a three-layer GAT model. Both of the
# first two layers consist of K = 4 attention heads computing F' = 256 features (for a total of 1024
# features), followed by an ELU nonlinearity.
# The final layer is used for (multi-label) classification:
# K = 6 attention heads computing 121 features each, that are averaged and followed by a logistic
# sigmoid activation.
# The training sets for this task are sufficiently large and we found no need to apply
# L2 regularization or dropout—we have, however,
# successfully employed skip connections (He et al., 2016) across the intermediate attentional layer.
class GAT(torch.nn.Module):
    def __init__(self, num_vocab, max_seq_len, node_encoder, emb_dim, inner_dim=128, num_layers=3, heads=4, num_class=0):
        super(GAT, self).__init__()

        self.num_class = num_class  # if we do classification
        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.node_encoder = node_encoder
        inner_dim = emb_dim  # TODO try also with 128

        self.heads=heads

        self.conv1 = GATConv(emb_dim, inner_dim, heads=heads)
        self.convs = nn.ModuleList(
            [GATConv(heads * inner_dim, inner_dim, heads=heads) for _ in range(num_layers - 2)])
        self.conv3 = GATConv(heads * inner_dim, 121, heads=6, concat=True)

        if self.num_class > 0:
            self.graph_pred_linear = torch.nn.Linear(121, self.num_class)
        else:

            self.graph_pred_linear_list = torch.nn.ModuleList()
            if self.num_vocab == 1:
                self.graph_pred_linear_list.append(torch.nn.Sequential(
                    torch.nn.Linear(121, self.num_vocab), torch.nn.ReLU()))
            else:
                for i in range(max_seq_len):
                    self.graph_pred_linear_list.append(torch.nn.Linear(121, self.num_vocab))


    def forward(self, batched_data):
        x, edge_index, node_depth, batch = batched_data.x, batched_data.edge_index,  batched_data.node_depth, batched_data.batch

        x = self.node_encoder(x, node_depth.view(-1,))

        # x = F.dropout(data.x, p=0.6, training=self.training)    # in their transductive tasks they use dropout...
        x = F.elu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)  # in their transductive tasks they use dropout...
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch.long())  # TODO it's not clear from the paper/code what they use here (sum/mean)
        x = torch.mean(x.view(-1, 6, 121), dim=1)

        if self.num_class > 0:
            return self.graph_pred_linear(x)

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](x))

        return pred_list


class SAGPoolGNN(nn.Module):

    def __init__(self, num_vocab, max_seq_len, node_encoder, emb_dim, pooling_ratio=0.5, dropout_ratio=0.5, num_layers=3, num_class=0):
        super(SAGPoolGNN, self).__init__()

        self.num_class = num_class
        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.node_encoder = node_encoder

        if self.num_class > 0:  # classification
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()

            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(self.emb_dim, self.num_vocab))

        # SAGPool original part

        self.num_features = emb_dim
        self.nhid = emb_dim
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.num_layers = num_layers

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.convs = nn.ModuleList([GCNConv(self.nhid, self.nhid) for _ in range(num_layers - 1)])
        self.pools = nn.ModuleList([SAGPooling(self.nhid, ratio=self.pooling_ratio) for _ in range(num_layers-1)])

        # self.conv1 = GCNConv(self.num_features, self.nhid)
        # self.conv2 = GCNConv(self.nhid, self.nhid)
        # self.conv3 = GCNConv(self.nhid, self.nhid)
        #
        # self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        # self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        # self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid)


    def forward(self, batched_data):
        x, edge_index, edge_attr, node_depth, batch = batched_data.x, batched_data.edge_index,  batched_data.edge_attr, batched_data.node_depth, batched_data.batch

        x = self.node_encoder(x, node_depth.view(-1,))

        xs = []
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        xs += [torch.cat([gmp(x, batch), gap(x, batch)], dim=1)]

        for i in range(self.num_layers-1):
            x = F.relu(self.convs[i](x, edge_index))
            x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, batch=batch)
            xs += [torch.cat([gmp(x, batch), gap(x, batch)], dim=1)]

        x = xs[0]
        for i in range(1, len(xs)):
            x += xs[i]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))

        if self.num_class > 0:
            return self.graph_pred_linear(x)

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](x))

        return pred_list
