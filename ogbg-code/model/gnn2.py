import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, SAGPooling, GraphUNet, DenseSAGEConv, dense_diff_pool, GatedGraphConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_sort_pool, global_mean_pool, global_add_pool
from torch_geometric.utils import dropout_adj, to_dense_batch, to_dense_adj
from math import ceil
import numpy as np
from model.conv import GCNConv as GCNConvOGB
from torch_geometric.nn import GatedGraphConv, GATConv
from torch_scatter import scatter_add
from tg.gated_graph_conv import GatedGraphConv as GatedGraphConv_EType

class GGNN_Simple(nn.Module):

    def __init__(self, num_vocab, max_seq_len, node_encoder, emb_dim,
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


# https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf
# code is adaptation of original code, used pytorch geometric where possible
# https://github.com/muhanzhang/pytorch_DGCNN/blob/master/DGCNN_embedding.py
#
# DGCNN has three sequential stages: 1) graph convolution
# layers extract vertices’ local substructure features and define
# a consistent vertex ordering; 2) a SortPooling layer sorts the
# vertex features under the previously defined order and unifies
# input sizes; 3) traditional convolutional and dense layers read
# the sorted graph representations and make predictions.
#
# The network has four graph convolution layers with 32, 32, 32, 1 output channels, respectively.
# For convenience, we set the last graph convolution to have one channel and only used this single channel for sorting.
# We set the k of SortPooling such
# that 60% graphs have nodes more than k. The remaining
# layers consist of two 1-D convolutional layers and one dense layer.
# The first 1-D convolutional layer has 16 output channels followed by a MaxPooling layer with filter size 2 and
# step size 2. The second 1-D convolutional layer has 32 output
# channels, filter size 5 and step size 1. The dense layer has
# 128 hidden units followed by a softmax layer as the output
# layer. A dropout layer with dropout rate 0.5 is used after the
# dense layer. We used the hyperbolic tangent function (tanh)
# as the nonlinear function in graph convolution layers, and rectified linear units (ReLU) in other layers
#
# see default params in https://github.com/muhanzhang/pytorch_DGCNN/blob/master/util.py and in model below
# cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
# cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
# out dim is outdim from DGCNN pooling but real classfification only comes thereafter
class DGCNN(nn.Module):
    def __init__(self, num_vocab, max_seq_len, node_encoder, emb_dim,
                 num_edge_feats, num_layers=3, #out_dim=1024, hidden_dim=100, num_classes=2,
                 k=30, init_gnn=True, # latent_dim=[32, 32, 32, 1],
                 conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='ReLU'):
        print('Initializing DGCNN')
        super(DGCNN, self).__init__()

        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.node_encoder = node_encoder
        self.hidden_dim = 32  # original value everything else like 128 gave memory errors
        # DGCNN original part
        latent_dim = [self.hidden_dim for _ in range(num_layers)] + [1]
        self.latent_dim = latent_dim
        # self.output_dim = out_dim
        self.num_node_feats = emb_dim
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        # if word_vectors is not None:
        #     self.emb = nn.EmbeddingBag.from_pretrained(word_vectors, freeze=True, mode=mode)
        # else:
        #     self.emb = None
        if init_gnn: # save space in unet
            self.conv_params = nn.ModuleList()
            self.conv_params.append(GCNConv(emb_dim + num_edge_feats, latent_dim[0]))
            for i in range(1, len(latent_dim)):
                self.conv_params.append(GCNConv(latent_dim[i-1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        # if out_dim > 0:
        #     self.out_params = nn.Linear(self.dense_dim, out_dim)
        #
        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))
        #
        # # added here directly from DGCNN's main
        # self.mlp = MLPClassifier(input_size=out_dim, hidden_size=hidden_dim, num_class=num_classes,
        #                          with_dropout=True, activation=activation)  # in paper, they say they use it, although not in default params

        self.graph_pred_linear_list = torch.nn.ModuleList()

        for i in range(max_seq_len):
            self.graph_pred_linear_list.append(torch.nn.Linear(self.dense_dim, self.num_vocab))

        weights_init(self)

    def forward(self, data):
        return self.sortpooling_embedding(data)

    def sortpooling_embedding(self, batched_data):
        x, edge_index, node_depth, batch = batched_data.x, batched_data.edge_index,  batched_data.node_depth, batched_data.batch

        x = self.node_encoder(x, node_depth.view(-1,))

        node_feat = x
        edge_feat = batched_data.edge_attr if hasattr(batched_data, 'edge_attr') and \
                                                                         batched_data.edge_attr is not None else None
        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            # we added inverse before..
            # edge_index = torch.cat([edge_index, torch.stack([edge_index[1],edge_index[0]], dim=0)], dim=-1)
            e2n_sp = torch.zeros(x.shape[0], edge_index.shape[1]).to(edge_feat.device).scatter_(0, edge_index, 1)
            e2npool_input = torch.mm(e2n_sp, edge_feat)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        cur_message_layer = self.compute_message_layers(node_feat, edge_index, batch)  # put in extra function to reuse rest in unet

        ''' sortpooling layer '''
        batch_sortpooling_graphs = global_sort_pool(cur_message_layer, batch, self.k)

        ''' traditional 1d convlution and dense layers '''
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = self.conv1d_activation(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = self.conv1d_activation(conv1d_res)

        to_dense = conv1d_res.view(conv1d_res.shape[0], -1)

        # if self.output_dim > 0:
        #     out_linear = self.out_params(to_dense)
        #     reluact_fp = self.conv1d_activation(out_linear)
        # else:
        #     reluact_fp = to_dense
        #
        # return self.mlp(self.conv1d_activation(reluact_fp))
        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](to_dense))

        return pred_list


    def compute_message_layers(self, feat, edge_index, batch=None):
        lv = 0
        cur_message_layer = feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            normalized_linear = self.conv_params[lv](cur_message_layer, edge_index)
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)
        return cur_message_layer


# https://github.com/muhanzhang/pytorch_DGCNN/blob/master/lib/pytorch_util.py
def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name:  # top-level parameters
            _param_init(p)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)


def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


# used for classification in main
# https://github.com/muhanzhang/pytorch_DGCNN/blob/f943f7e920e8b73384c2ad9610110b855632b42c/mlp_dropout.py#L46
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False, activation=None):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        self.activation = init_activation(activation, num_class)

        weights_init(self)

    def forward(self, x):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = self.activation(logits)
        return logits


# https://arxiv.org/pdf/1806.08804.pdf
# code parts see
# https://github.com/RexYing/diffpool/blob/8dfb97cf60c2376ac804761837b9966f1d302acb/encoders.py (default params in train.py)
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/proteins_diff_pool.py
#
# Model configurations. In our experiments, the GNN model used for DIFFPOOL is built on top of
# the GRAPHSAGE architecture, as we found this architecture to have superior performance compared
# to the standard GCN approach as introduced in [22].
# We use the “mean” variant of GRAPHSAGE [16]
# and apply a DIFFPOOL layer after every two GRAPHSAGE layers in our architecture. A total of 2
# DIFFPOOL layers are used for the datasets. ...
# After each DIFFPOOL layer, 3 layers of graph convolutions are performed,
# before the next DIFFPOOL layer, or the readout layer.
# The embedding matrix and the assignment matrix are computed by two separate GRAPHSAGE models respectively.
# In the 2 DIFFPOOL layer architecture, the number of clusters is set as 25% of the number of nodes
# before applying DIFFPOOL, while in the 1 DIFFPOOL layer architecture, the number of clusters is set
# as 10%. Batch normalization [18] is applied after every layer of GRAPHSAGE. We also found that
# adding an `2 normalization to the node embeddings at each layer made the training more stable.
# code defult parameters
#                         feature_type='default',
#                         batch_size=20,
#                         input_dim=10,
#                         hidden_dim=20, slightly small? -- not using those since other datasets
#                         output_dim=20, (our embedding_dim)
#                         num_classes=2,
#                         num_gc_layers=3, (automatically in our GNN)
#                         dropout=0.0,
#                         method='base',
#                         num_pool=1
class DiffPoolGNN(nn.Module):
    def __init__(self, num_vocab, max_seq_len, node_encoder, emb_dim,
                 #in_dim, hidden_dim, embedding_dim, out_dim, word_vectors=None, mode='sum', activation=None,
                 max_nodes=1000):
        super(DiffPoolGNN, self).__init__()

        # if word_vectors is not None:
        #     self.emb = nn.EmbeddingBag.from_pretrained(word_vectors, freeze=True, mode=mode)
        # else:
        #     self.emb = None
        in_dim = emb_dim
        hidden_dim = emb_dim
        embedding_dim = emb_dim

        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.node_encoder = node_encoder

        self.graph_pred_linear_list = torch.nn.ModuleList()

        for i in range(max_seq_len):
            self.graph_pred_linear_list.append(torch.nn.Linear(self.emb_dim, self.num_vocab))

        self.gnn1_embed = GNN(in_dim, hidden_dim, embedding_dim, lin=False)

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(in_dim, hidden_dim, num_nodes)

        self.gnn2_embed = GNN(2 * hidden_dim + embedding_dim, hidden_dim, embedding_dim, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(2*hidden_dim+embedding_dim, hidden_dim, num_nodes)

        self.gnn3_embed = GNN(2*hidden_dim+embedding_dim, hidden_dim, embedding_dim, lin=False)

        self.lin1 = torch.nn.Linear(2*hidden_dim+embedding_dim, hidden_dim)
        # self.lin2 = torch.nn.Linear(hidden_dim, out_dim)

        # self.activation = init_activation(activation, out_dim)

    def forward(self, batched_data, mask=None):

        x, edge_index, edge_attr, node_depth, batch = batched_data.x, batched_data.edge_index,  batched_data.edge_attr, batched_data.node_depth, batched_data.batch

        x = self.node_encoder(x, node_depth.view(-1,))

        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        # x = self.lin2(x)
        # return self.activation(x)  #, l1 + l2, e1 + e2

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](x))

        return pred_list

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):  #add_loop=False,
        super(GNN, self).__init__()

        # self.add_loop = add_loop
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


# https://arxiv.org/pdf/1905.05178.pdf
# https://github.com/HongyangGao/Graph-U-Nets/blob/master/network.py is nearly entirely DGCNN!
# just one UNetLayer -- using pytorch-geometric's for that
#
# For inductive learning tasks, we follow the same experimental setups in (Zhang et al., 2018) using our g-U-Nets
# architecture as described in transductive learning settings for feature extraction.
# Since the sizes of graphs vary in graph classification tasks, we sample proportions of nodes in four
# gPool layers; those are 90%, 70%, 60%, and 50%, respectively.
# The dropout keep rate imposed on feature matrices is 0.3. => 0.7 dropout-p
#
# For transductive learning tasks, we employ our proposed g-U-Nets proposed in Section 3.3.
# Since nodes in the three datasets are associated with high-dimensional features, we employ a GCN layer to reduce them
# into low-dimensional representations. (is in tg UNet!)
# In the encoder part, we stack four blocks, each of which consists of a gPool layer and a GCN layer. We sample  ...
# Finally, we apply a GCN layer for final prediction. For all layers in the
# model, we use identity activation function (Gao et al., 2018)
# after each GCN layer.
# To avoid over-fitting, we apply L2 regularization on weights with λ = 0.001.
# Dropout (Srivastava et al., 2014) is applied to both adjacency matrices
# and feature matrices with keep rates of 0.8 and 0.08, respectively.
class UNet(DGCNN):
    def __init__(self, num_vocab, max_seq_len, node_encoder, emb_dim,
                 num_edge_feats, num_layers, #out_dim=1024, hidden_dim=100, num_classes=2,
                 k=30):
        super(UNet, self).__init__(num_vocab, max_seq_len, node_encoder, emb_dim, num_edge_feats, 4, k=k, init_gnn=False)

        self.unet = GraphUNet(emb_dim+num_edge_feats, emb_dim, self.total_latent_dim, depth=4, pool_ratios=[0.9, 0.7, 0.6, 0.5])

    # def forward(self, data):
    #     edge_index, _ = dropout_adj(data.edge_index, p=0.2,
    #                                 force_undirected=True,
    #                                 num_nodes=data.num_nodes,
    #                                 training=self.training)
    #     x = F.dropout(data.x, p=0.7, training=self.training)
    #
    #     TBD set in data or do above in below method later?
    #     self.sortpooling_embedding(data)

    def compute_message_layers(self, feat, edge_index, batch=None):
        # in original code they normalize A here, I assume not needed since tg UNet uses GCNConv
        return self.unet(feat, edge_index, batch)

    # simpler alternative
    # def forward(self, data):
    #     edge_index, _ = dropout_adj(data.edge_index, p=0.2,
    #                                 force_undirected=True,
    #                                 num_nodes=data.num_nodes,
    #                                 training=self.training)
    #     x = F.dropout(data.x, p=0.7, training=self.training)
    #
    #     x = self.unet(x, edge_index)
    #     x = self.gcn_pred(x, edge_index)
    #
    #     x = gmp(x, batch=data.batch)  # we need some pooling? TODO
    #     x = self.relu(self.lin(x))
    #
    #     return self.activation(x)


# code from https://github.com/inyeoplee77/SAGPool/blob/master/networks.py mixed with tg
#
# Hierarchical pooling architecture In this setting, we implemented the hierarchical pooling architecture from the
# recent hierarchical pooling study of Cangea et al.. As shown
# in Figure 2, the architecture is comprised of three blocks
# each of which consists of a graph convolutional layer and
# a graph pooling layer. The outputs of each block are summarized in the readout layer. The summation of the outputs
# of each readout layer is fed to the linear layer for classification
#
# Since the global pooling architecture ...
# minimizes the loss of information, it performs better than
# the hierarchical pooling architecture P OOLh (SAGPoolh,
# gPoolh, DiffPoolh) on datasets with fewer nodes (NCI1,
# NCI109, FRANKENSTEIN). However, P OOLh is more
# effective on datasets with a large number of nodes (D&D,
# PROTEINS) because it efficiently extracts useful information from large scale graphs. T
# parser.add_argument('--seed', type=int, default=777,
#                     help='seed')
# parser.add_argument('--batch_size', type=int, default=128,
#                     help='batch size')
# parser.add_argument('--lr', type=float, default=0.0005,
#                     help='learning rate')
# parser.add_argument('--weight_decay', type=float, default=0.0001,
#                     help='weight decay')
# parser.add_argument('--nhid', type=int, default=128,
#                     help='hidden size')
# parser.add_argument('--epochs', type=int, default=100000,
#                     help='maximum number of epochs')
# parser.add_argument('--patience', type=int, default=50,
#                     help='patience for earlystopping')
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


class SAGPoolGNN_EA(nn.Module):

    def __init__(self, num_vocab, max_seq_len, node_encoder, emb_dim, pooling_ratio=0.5, dropout_ratio=0.5, num_layers=3):
        super(SAGPoolGNN_EA, self).__init__()

        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.node_encoder = node_encoder

        self.graph_pred_linear_list = torch.nn.ModuleList()

        for i in range(max_seq_len):
            self.graph_pred_linear_list.append(torch.nn.Linear(self.emb_dim, self.num_vocab))

        # SAGPool original part

        self.num_features = emb_dim
        self.nhid = emb_dim
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.num_layers = num_layers

        self.conv1 = GCNConvOGB(self.emb_dim) #num_features, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.convs = nn.ModuleList([GCNConvOGB(self.emb_dim) for _ in range(num_layers - 1)])
            # self.nhid, self.nhid) for _ in range(num_layers - 1)])
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
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        xs += [torch.cat([gmp(x, batch), gap(x, batch)], dim=1)]

        for i in range(self.num_layers-1):
            x = F.relu(self.convs[i](x, edge_index, edge_attr))
            x, edge_index, edge_attr, batch, _, _ = self.pools[i](x, edge_index, edge_attr, batch)
            xs += [torch.cat([gmp(x, batch), gap(x, batch)], dim=1)]

        x = xs[0]
        for i in range(1, len(xs)):
            x += xs[i]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](x))

        return pred_list



def init_activation(activation, out_dim):
    if activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softmax":
        return nn.Softmax(dim=-1)
    else:
        return nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=-1)
