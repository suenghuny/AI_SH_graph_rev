import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import sys
sys.path.append("..")  # 상위 폴더를 import할 수 있도록 경로 추가
from cfg import get_cfg
from GTN.inits import glorot
cfg = get_cfg()
print(torch.cuda.device_count())
device =torch.device(cfg.cuda if torch.cuda.is_available() else "cpu")
print(device)

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, n_representation_obs, layers = [20, 30 ,40]):
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.linears = OrderedDict()
        last_layer = self.feature_size
        for i in range(len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.linears['linear{}'.format(i)]= nn.Linear(last_layer, layer)
                self.linears['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.linears['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            else:
                self.linears['linear{}'.format(i)] = nn.Linear(last_layer, n_representation_obs)

        self.node_embedding = nn.Sequential(self.linears)
        #print(self.node_embedding)
        self.node_embedding.apply(weight_init_xavier_uniform)


    def forward(self, node_feature, missile=False):
        #print(node_feature.shape)
        node_representation = self.node_embedding(node_feature)
        return node_representation

class HGNN(nn.Module):
    def __init__(self, feature_size, embedding_size, graph_embedding_size, layers, num_node_cat, num_edge_cat):
        super(HGNN, self).__init__()
        embedding_layers = [NodeEmbedding(feature_size, embedding_size, layers) for _ in range(num_node_cat)]
        self.embedding_layers = nn.ModuleList(embedding_layers)


        self.Ws = []
        for i in range(num_edge_cat):
            self.Ws.append(nn.Parameter(torch.Tensor(embedding_size, graph_embedding_size)))
        self.Ws = nn.ParameterList(self.Ws)
        [glorot(W) for W in self.Ws]

    #def forward(self, A, X, num_nodes=None, mini_batch=False):

    def forward(self, A, X, mini_batch, layer = 0):
        if mini_batch == False:
            for E in range(len(A)):
                num_nodes = X.shape[0]
                print(torch.ones(torch.tensor(E).shape[0]))
                E = torch.sparse_coo_tensor(E, torch.ones(torch.tensor(E).shape[0]), (num_nodes, num_nodes)).to(device).to_dense()
                print(X@E.shape)

            print(X.shape, len(A))
        else:
            mat_a = [torch.zeros(self.num_edge_type, num_nodes, num_nodes).to(device) for _ in range(batch_size)]
            for b in range(batch_size):
                for i in range(self.num_edge_type):

                    mat_a[b][i].copy_(torch.sparse_coo_tensor(A[b][i][0], A[b][i][1], (num_nodes, num_nodes)).to(device).to_dense())
            mat_a = torch.stack(mat_a, dim=0)
            Hs = torch.einsum('bcij, bcjk-> bcik', torch.einsum('bijk, ci  -> bcjk', mat_a, filter), H_)

            # empty = list()
            # for i in range(len(node_cats)):
            #     if i == 0:
            #         start_idx = 0
            #         end_idx = node_cats[0]
            #     else:
            #         start_idx = node_cats[i-1]+1
            #         end_idx = node_cats[i]
            #     embedding = self.embedding_layers(X[start_idx:end_idx, :])
            #     empty.append(embedding)
            # torch.cat(empty, dim = 1)



