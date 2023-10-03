import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
# from torch_geometric.nn import SAGEConv
import dgl
from dgl.nn import DenseGraphConv

from .embedding_update import MLPUpdater, GRUUpdater
# from models.args import get_args

# args = get_args()
# device = torch.device('cuda', args.local_rank)


class DynamicGNN(nn.Module):
    r"""
    SAGConv: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
    dgl DenseGraphConv: https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.DenseGraphConv.html
    """
    def __init__(self, dim_in: int, dim_out: int, adj: torch.tensor, embed_update: str = 'GRU'):
        super(DynamicGNN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out)
        self.agg_linear = nn.Linear(dim_in + dim_out, dim_in)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(dim_out)
        self.drop = nn.Dropout(p=0.9)
        self.adj = adj
        self.embed_update = embed_update
        

    def forward(self, node_feature, prev_node_feature):
        r""" 
        node_feature shape: N * dim (N: number of nodes)
        prev_node_feature shape: N * dim (N: number of nodes)
        """
        # gnn = DenseGraphConv(node_feature.shape[1], node_feature.shape[1]).to(device)

        # if self.embed_update == 'GRU':
        #     embd_up = GRUUpdater(node_feature.shape[1], node_feature.shape[1]).to(device)
        # elif self.embd_up == 'MLP':
        #     embd_up = MLPUpdater(node_feature.shape[1], node_feature.shape[1]).to(device)
        # else:
        #     raise 'Wrong mode!'

        gnn = DenseGraphConv(node_feature.shape[1], node_feature.shape[1]).cuda()

        if self.embed_update == 'GRU':
            embd_up = GRUUpdater(node_feature.shape[1], node_feature.shape[1])
        elif self.embd_up == 'MLP':
            embd_up = MLPUpdater(node_feature.shape[1], node_feature.shape[1])
        else:
            raise 'Wrong mode!'

        # print("adj: ", self.adj)
        # print("node feature: ", node_feature)
        res = gnn(self.adj, node_feature) 
        res1 = res + node_feature
        res2 = embd_up(prev_node_feature, res1)
        res3 = gnn(self.adj, res2)
        res4 = res3 + res2
        res5 = embd_up(prev_node_feature, res4)
        out = res5
        print("Dynamic GNN running ...")

        # node_feature = SAGConv(node_feature)
        # prev_node_feature = SAGConv(prev_node_feature)
        return out

if __name__ == '__main__':
    r"""
    Test code.
    """
    curr_fea = torch.randn((5, 32))
    prev_fea = torch.randn((5, 32))
    print(curr_fea)
    d_in = curr_fea.shape[1]
    d_out = prev_fea.shape[1]
    adj = torch.ones((curr_fea.shape[0], curr_fea.shape[0]))
    print(adj)
    model = DynamicGNN(d_in, d_out, adj)
    fea = model(curr_fea, prev_fea)
    print("Updated feature: ", fea)
    print(fea.shape) # torch.Size([5, 32])

