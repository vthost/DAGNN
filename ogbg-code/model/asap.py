import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (ASAPooling,
                                GraphConv, global_mean_pool,
                                JumpingKnowledge)


class ASAP(torch.nn.Module):
    def __init__(self, num_vocab, max_seq_len, node_encoder, emb_dim, num_layers, hidden, ratio=0.8, dropout=0, num_class=0):
        super(ASAP, self).__init__()

        self.num_class = num_class
        self.max_seq_len = max_seq_len
        self.node_encoder = node_encoder

        self.conv1 = GraphConv(emb_dim, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(hidden, ratio, dropout=dropout)
            for i in range((num_layers) // 2)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)

        if self.num_class > 0:  # classification
            self.graph_pred_linear = torch.nn.Linear(hidden, self.num_class)
        else:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(hidden, num_vocab))

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, node_depth, batch = data.x, data.edge_index, data.node_depth, data.batch

        x = self.node_encoder(x, node_depth.view(-1, ))

        edge_weight = None
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=-1)

        if self.num_class > 0:
            return self.graph_pred_linear(x)

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](x))
        return pred_list

    def __repr__(self):
        return self.__class__.__name__

