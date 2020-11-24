
import os
import torch
import statistics

class ASTNodeEncoder2(torch.nn.Module):
    '''
        Input:
            x: default node feature. the first and second column represents node type and node attributes.
            depth: The depth of the node in the AST.

        Output:
            emb_dim-dimensional vector

    '''
    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super(ASTNodeEncoder2, self).__init__()

        self.max_depth = max_depth

        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        # self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)


    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:,0]) + self.attribute_encoder(x[:,1]) #+ self.depth_encoder(depth)


def augment_edge2(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    '''

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    # edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim = 0)
    # edge_attr_ast_inverse = torch.cat([torch.zeros(edge_index_ast_inverse.size(1), 1), torch.ones(edge_index_ast_inverse.size(1), 1)], dim = 1)


    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(data.node_is_attributed.view(-1,) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]], dim = 0)
    edge_attr_nextoken = torch.cat([torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim = 1)


    ##### Inverse next-token edge
    # edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim = 0)
    # edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))


    data.edge_index = torch.cat([edge_index_ast, edge_index_nextoken], dim = 1)
    data.edge_attr = torch.cat([edge_attr_ast, edge_attr_nextoken], dim = 0)

    return data


def summary_report(val_list):
    return sum(val_list)/len(val_list), statistics.stdev(val_list) if len(val_list) > 1 else 0


def create_checkpoint(checkpoint_fn, epoch, model, optimizer, results):
    checkpoint = {"epoch": epoch,
                  "model": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "results": results}
    torch.save(checkpoint, checkpoint_fn)


def remove_checkpoint(checkpoint_fn):
    os.remove(checkpoint_fn)


def load_checkpoint(checkpoint_fn, model, optimizer):
    checkpoint = torch.load(checkpoint_fn)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['results'], checkpoint['epoch'], model, optimizer


def load_checkpoint_results(checkpoint_fn):
    checkpoint = torch.load(checkpoint_fn)
    return checkpoint['results']

