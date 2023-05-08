from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from torch.distributions import Categorical
import numpy as np

from GAT.model import GAT
from GAT.layers import device
from copy import deepcopy
from GTN.utils import _norm
from GTN.model_fastgtn import FastGTNs
from scipy.sparse import csr_matrix
from collections import OrderedDict
from NoisyLinear import NoisyLinear

from cfg import get_cfg

cfg = get_cfg()

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

class IQN(nn.Module):
    def __init__(self, state_size, action_size, batch_size, layer_size=128, N=12, layers = [128, 96, 64, 56, 48], n_cos = 64):
        super(IQN, self).__init__()
        self.input_shape = state_size
        self.batch_size = batch_size
        # print(state_size)
        self.action_size = action_size
        self.N = N
        self.n_cos = n_cos
        self.layer_size = layer_size
        #print(self.N, self.n_cos)
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(device)

        self.head = nn.Linear(self.input_shape, layer_size)  # cound be a cnn
        self.head_y = nn.Linear(self.input_shape, layer_size)  # cound be a cnn
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)



        self.noisylinears_for_advantage = OrderedDict()
        last_layer = layer_size
        for i in range(len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                if cfg.epsilon_greedy == False:
                    self.noisylinears_for_advantage['linear{}'.format(i)]= NoisyLinear(last_layer, layer)
                else:
                    self.noisylinears_for_advantage['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.noisylinears_for_advantage['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.noisylinears_for_advantage['activation{}'.format(i)] = nn.ReLU()
                last_layer = layer
            else:
                if cfg.epsilon_greedy == False:
                    self.noisylinears_for_advantage['linear{}'.format(i)] = NoisyLinear(last_layer, self.action_size)
                else:
                    self.noisylinears_for_advantage['linear{}'.format(i)] = nn.Linear(last_layer, self.action_size)
        #print("/&/&/&",self.noisylinears_for_advantage)
        self.noisylinears_for_v = OrderedDict()
        last_layer = layer_size
        for i in range(len(layers)):
            layer = layers[i]
            if i <= len(layers) - 2:
                print(cfg.epsilon_greedy)
                if cfg.epsilon_greedy == False:
                    self.noisylinears_for_v['linear{}'.format(i)]= NoisyLinear(last_layer, layer)
                else:
                    self.noisylinears_for_v['linear{}'.format(i)] = nn.Linear(last_layer, layer)

                self.noisylinears_for_v['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.noisylinears_for_v['activation{}'.format(i)] = nn.ReLU()
                last_layer = layer
            else:
                if cfg.epsilon_greedy == False:
                    self.noisylinears_for_v['linear{}'.format(i)] = NoisyLinear(last_layer, 1)
                else:
                    self.noisylinears_for_v['linear{}'.format(i)] = nn.Linear(last_layer, 1)
        self.advantage_layer = nn.Sequential(self.noisylinears_for_advantage)
        print(self.advantage_layer)
        self.v_layer = nn.Sequential(self.noisylinears_for_v)
        if cfg.epsilon_greedy == False:
            self.reset_noise_net()


    def reset_noise_net(self):
        for layer in self.v_layer:
            if type(layer) is NoisyLinear:
                layer.sample_noise()
        for layer in self.advantage_layer:
            if type(layer) is NoisyLinear:
                layer.sample_noise()

    def remove_noise_net(self):
        for layer in self.v_layer:
            if type(layer) is NoisyLinear:
                layer.remove_noise()
        for layer in self.advantage_layer:
            if type(layer) is NoisyLinear:
                layer.sample_noise().remove_noise()



    def calc_cos(self, batch_size):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, self.N).to(device).unsqueeze(-1)  # (batch_size, self.N, 1)
        cos = torch.cos(taus * self.pis)  # self.pis shape : 1,1,self.n_cos
        assert cos.shape == (batch_size, self.N, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward(self, input, cos, mini_batch):



        N = self.N
        """
        Quantile Calculation depending on the number of tau
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        """
        if mini_batch == False:
            batch_size = 1
            #input = input.unsqueeze(0)
        else:
            batch_size = self.batch_size

        x = torch.relu(self.head(input.to(device)))  # x의 shape는 batch_size, layer_size
        cos = cos.view(batch_size * N, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, N, self.layer_size)  # (batch, n_tau, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * N, self.layer_size)  # 이부분이 phsi * phi에 해당하는 부분
        out_a = self.advantage_layer(x)
        quantiles_a = out_a.view(batch_size, N, self.action_size)
        a = quantiles_a.mean(dim=1)

        y = torch.relu(self.head_y(input.to(device)))  # x의 shape는 batch_size, layer_size
        cos_y = cos_x
        y = (y.unsqueeze(1) * cos_y).view(batch_size * N, self.layer_size)  # 이부분이 phsi * phi에 해당하는 부분
        out_v = self.v_layer(y)
        quantiles_v = out_v.view(batch_size, N, 1)
        v = quantiles_v.mean(dim=1)

       # print(quantiles_v.shape, quantiles_a.shape, a.shape)

        q = v + a-a.mean(dim= 1, keepdims = True)
        return q


class VDN(nn.Module):

    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_local):
        return torch.sum(q_local, dim=1)


class Network(nn.Module):
    def __init__(self, obs_and_action_size, hidden_size_q, action_size):
        super(Network, self).__init__()
        self.obs_and_action_size = obs_and_action_size
        self.fcn_1 = nn.Linear(obs_and_action_size, hidden_size_q + 10)
        self.fcn_1bn = nn.BatchNorm1d(hidden_size_q + 10)

        self.fcn_2 = nn.Linear(hidden_size_q + 10, hidden_size_q - 5)
        self.fcn_2bn = nn.BatchNorm1d(hidden_size_q - 5)

        self.fcn_3 = nn.Linear(hidden_size_q - 5, hidden_size_q - 20)
        self.fcn_3bn = nn.BatchNorm1d(hidden_size_q - 20)

        self.fcn_4 = nn.Linear(hidden_size_q - 20, hidden_size_q - 40)
        self.fcn_4bn = nn.BatchNorm1d(hidden_size_q - 40)

        self.fcn_5 = nn.Linear(hidden_size_q - 40, action_size)
        # self.fcn_5 = nn.Linear(int(hidden_size_q/8), action_size)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)
        torch.nn.init.xavier_uniform_(self.fcn_4.weight)
        torch.nn.init.xavier_uniform_(self.fcn_5.weight)
        # torch.nn.init.xavier_uniform_(self.fcn_5.weight)

    def forward(self, obs_and_action):
        if obs_and_action.dim() == 1:
            obs_and_action = obs_and_action.unsqueeze(0)
        # print(obs_and_action.dim())

        x = F.elu(self.fcn_1bn(self.fcn_1(obs_and_action)))
        x = F.elu(self.fcn_2bn(self.fcn_2(x)))
        x = F.elu(self.fcn_3bn(self.fcn_3(x)))
        x = F.elu(self.fcn_4bn(self.fcn_4(x)))
        q = self.fcn_5(x)
        # q = self.fcn_5(x)
        return q


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
                self.linears['activation{}'.format(i)] = nn.ReLU()
                last_layer = layer
            else:
                self.linears['linear{}'.format(i)] = nn.Linear(last_layer, n_representation_obs)

        self.node_embedding = nn.Sequential(self.linears)
       # print(self.node_embedding)
        self.node_embedding.apply(weight_init_xavier_uniform)


    def forward(self, node_feature, missile=False):

        node_representation = self.node_embedding(node_feature)
        return node_representation


class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, n_node_feature_missile, n_node_feature_enemy, action_size, n_step, per_alpha):
        self.buffer = deque()
        self.alpha = per_alpha
        self.step_count_list = list()
        for _ in range(13):
            self.buffer.append(deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.n_node_feature_missile = n_node_feature_missile
        self.n_node_feature_enemy = n_node_feature_enemy
        #self.agent_id = np.eye(self.num_agent).tolist()
        self.one_hot_actions = np.eye(action_size).tolist()
        self.batch_size = batch_size
        self.step_count = 0
        self.rewards_store = list()
        self.n_step = n_step

    def pop(self):
        self.buffer.pop()

    def memory(self,
               node_feature_missile,
               ship_feature,
               edge_index_missile,
               action,
               reward,
               done,
               avail_action,
               node_feature_enemy,
               edge_index_enemy, status
               ):

        self.buffer[1].append(node_feature_missile)
        self.buffer[2].append(ship_feature)

        self.buffer[3].append(edge_index_missile)
        self.buffer[4].append(action)


        self.buffer[5].append(list(reward))
        self.buffer[6].append(list(done))


        self.buffer[7].append(avail_action)
        self.buffer[8].append(status)
        self.buffer[9].append(np.sum(status))

        if len(self.buffer[10]) == 0:
            self.buffer[10].append(0.5)
        else:
            self.buffer[10].append(max(self.buffer[10]))

        self.buffer[11].append(node_feature_enemy)
        self.buffer[12].append(edge_index_enemy)


        if self.step_count < self.buffer_size:
            self.step_count_list.append(self.step_count)
            self.step_count += 1

        #print(len(self.buffer[7]), len(self.buffer[10]), len(self.step_count_list))

#ship_features
    def generating_mini_batch(self, datas, batch_idx, cat):
        for s in batch_idx:
            if cat == 'node_feature_missile':
                yield datas[1][s]
            if cat == 'ship_features':
                yield datas[2][s]
            if cat == 'edge_index_missile':
                yield torch.sparse_coo_tensor(datas[3][s],
                                              torch.ones(torch.tensor(datas[3][s]).shape[1]),
                                              (self.n_node_feature_missile, self.n_node_feature_missile)).to_dense()
            if cat == 'action':
                yield datas[4][s]
            if cat == 'reward':
                yield datas[5][s]
            if cat == 'done':
                yield datas[6][s]
            if cat == 'node_feature_missile_next':
                yield datas[1][s + self.n_step]
            if cat == 'ship_features_next':
                yield datas[2][s + self.n_step]
            if cat == 'edge_index_missile_next':
                yield torch.sparse_coo_tensor(datas[3][s + self.n_step],
                                              torch.ones(torch.tensor(datas[3][s + self.n_step]).shape[1]),
                                              (self.n_node_feature_missile, self.n_node_feature_missile)).to_dense()
            if cat == 'avail_action_next':
                yield datas[7][s + self.n_step]
            if cat == 'status':
                yield datas[8][s]
            if cat == 'status_next':
                yield datas[8][s+ self.n_step]

            if cat == 'priority':
                yield datas[10][s]

            if cat == 'node_feature_enemy':
                yield datas[11][s]
            if cat == 'edge_index_enemy':
                yield torch.sparse_coo_tensor(datas[12][s],
                                              torch.ones(torch.tensor(datas[12][s]).shape[1]),
                                              (self.n_node_feature_enemy, self.n_node_feature_enemy)).to_dense()
            if cat == 'node_feature_enemy_next':
                yield datas[11][s]
            if cat == 'edge_index_enemy_next':
                yield torch.sparse_coo_tensor(datas[12][s + self.n_step],
                                              torch.ones(torch.tensor(datas[12][s + self.n_step]).shape[1]),
                                              (self.n_node_feature_enemy, self.n_node_feature_enemy)).to_dense()
    #def search_sample_space(self, sampled_batch_idx):

    def update_transition_priority(self, batch_index, delta):

        copied_delta_store = deepcopy(list(self.buffer[10]))
        delta = np.abs(delta) + np.min(copied_delta_store)
        priority = np.array(copied_delta_store).astype(float)
        batch_index = batch_index.astype(int)
        priority[batch_index] = delta
        #print("후", self.buffer[1][d])
        self.buffer[10]= deque(priority, maxlen=self.buffer_size)


    def sample(self, vdn):
        step_count_list = self.step_count_list[:]
        step_count_list.pop()

        if vdn == False:
            priority_point = list(self.buffer[9])[:]
            priority_point.pop()
            one_ratio = priority_point.count(1)/len(priority_point)
            #print(step_count_list)

            if np.random.uniform(0, 1) <= one_ratio:
                #print(len(step_count_list), (np.array(priority_point)/np.sum(priority_point)).shape)
                sampled_batch_idx =np.random.choice(step_count_list, p = np.array(priority_point)/np.sum(priority_point), size = self.batch_size)
            else:
                sampled_batch_idx = np.random.choice(step_count_list, size=self.batch_size)
        else:
            priority = list(deepcopy(self.buffer[10]))[:-self.n_step]
            p = np.array(priority)**self.alpha#/np.array(priority).sum()).tolist()
            p /= p.sum()
            p = p.tolist()
            #print(step_count_list[:-self.n_step+1], len(p), len(priority))

            sampled_batch_idx = np.random.choice(step_count_list[:-self.n_step+1], size=self.batch_size, p = p)

        node_feature_missile = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_missile')
        node_features_missile = list(node_feature_missile)

        action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action')
        actions = list(action)

        ship_features = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='ship_features')
        ship_features = list(ship_features)

        ship_features_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                                 cat='ship_features_next')
        ship_features_next = list(ship_features_next)

        edge_index_missile = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_missile')

        edge_indices_missile = list(edge_index_missile)

        reward = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='reward')
        rewards = list(reward)

        done = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='done')
        dones = list(done)

        #


        node_feature_missile_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                               cat='node_feature_missile_next')
        node_features_missile_next = list(node_feature_missile_next)



        edge_index_missile_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                             cat='edge_index_missile_next')
        edge_indices_missile_next = list(edge_index_missile_next)

        avail_action_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action_next')
        avail_actions_next = list(avail_action_next)

        status = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='status')
        status = list(status)


        status_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='status_next')
        status_next = list(status_next)

        priority = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='priority')
        priority = list(priority)

        node_feature_enemy = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_enemy')
        node_feature_enemy = list(node_feature_enemy)

        edge_index_enemy = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_enemy')
        edge_index_enemy = list(edge_index_enemy)

        node_feature_enemy_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_enemy_next')
        node_feature_enemy_next = list(node_feature_enemy_next)

        edge_index_enemy_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_enemy_next')
        edge_index_enemy_next = list(edge_index_enemy_next)

#

        return node_features_missile, ship_features, edge_indices_missile, actions, rewards, dones, node_features_missile_next, ship_features_next, edge_indices_missile_next, avail_actions_next, status, status_next,priority, sampled_batch_idx, node_feature_enemy, edge_index_enemy, node_feature_enemy_next, edge_index_enemy_next


class Agent:
    def __init__(self,
                 num_agent,
                 feature_size_ship,
                 feature_size_missile,
                 feature_size_enemy,

                 iqn_layers,

                 node_embedding_layers_ship,
                 node_embedding_layers_missile,
                 node_embedding_layers_enemy,

                 n_multi_head,
                 n_representation_ship,
                 n_representation_missile,
                 n_representation_enemy,





                 hidden_size_enemy,
                 hidden_size_comm,


                 dropout,
                 action_size,
                 buffer_size,
                 batch_size,
                 learning_rate,
                 gamma,
                 GNN,
                 teleport_probability,
                 gtn_beta,
                 n_node_feature_missile,
                 n_node_feature_enemy,
                 n_step,
                 beta,
                 per_alpha,
                 iqn_layer_size,
                 iqn_N,
                 n_cos

                 ):
        self.n_step = n_step
        self.num_agent = num_agent


        self.feature_size_ship = feature_size_ship
        self.feature_size_missile = feature_size_missile

        self.n_multi_head = n_multi_head
        self.teleport_probability = teleport_probability
        self.beta = beta

        self.n_representation_ship = n_representation_ship
        self.n_representation_missile = n_representation_missile
        self.n_representation_enemy = n_representation_enemy

        self.action_size = action_size

        self.dropout = dropout
        self.gamma = gamma
        self.agent_id = np.eye(self.num_agent).tolist()
        self.agent_index = [i for i in range(self.num_agent)]
        self.max_norm = 10
        self.VDN = VDN().to(device)
        self.VDN_target = VDN().to(device)

        self.VDN_target.load_state_dict(self.VDN.state_dict())
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = Replay_Buffer(self.buffer_size, self.batch_size, n_node_feature_missile,n_node_feature_enemy, self.action_size, n_step = self.n_step, per_alpha = per_alpha)
        self.n_node_feature_missile = n_node_feature_missile
        self.n_node_feature_enemy = n_node_feature_enemy

        self.action_space = [i for i in range(self.action_size)]

        self.GNN = GNN

        self.gamma_n_step = torch.tensor([[self.gamma**i for i in range(self.n_step+1)] for _ in range(self.batch_size)], dtype = torch.float, device = device)


        if self.GNN == 'GAT':

            self.node_representation_ship_feature = NodeEmbedding(feature_size=feature_size_ship,
                                                             n_representation_obs=n_representation_ship,
                                                             layers = node_embedding_layers_ship).to(device)  # 수정사항

            self.node_representation = NodeEmbedding(feature_size=feature_size_missile,
                                                     n_representation_obs=n_representation_missile,
                                                     layers = node_embedding_layers_missile).to(device)  # 수정사항

            self.node_representation_enemy = NodeEmbedding(feature_size=feature_size_enemy,
                                                     n_representation_obs=n_representation_enemy,
                                                     layers = node_embedding_layers_enemy).to(device)  # 수정사항


            self.func_missile_obs = GAT(nfeat=n_representation_missile,
                                         nhid=hidden_size_comm,
                                         nheads=n_multi_head,
                                         nclass=n_representation_missile+5,
                                         dropout=dropout,
                                         alpha=0.2,
                                         mode='observation',
                                         batch_size=self.batch_size,
                                         teleport_probability=self.teleport_probability).to(device)  # 수정사항


            self.func_enemy_obs = GAT(nfeat=n_representation_enemy,
                                         nhid=hidden_size_enemy,
                                         nheads=n_multi_head,
                                         nclass=n_representation_enemy+5,
                                         dropout=dropout,
                                         alpha=0.2,
                                         mode='observation',
                                         batch_size=self.batch_size,
                                         teleport_probability=self.teleport_probability).to(device)  # 수정사항

            self.Q = IQN(n_representation_ship+n_representation_missile + n_representation_enemy+10, self.action_size,
                         batch_size=self.batch_size, layer_size=iqn_layer_size, N=iqn_N, n_cos = n_cos, layers = iqn_layers).to(device)
            self.Q_tar = IQN(n_representation_ship+n_representation_missile + n_representation_enemy+10, self.action_size,
                             batch_size=self.batch_size, layer_size=iqn_layer_size, N=iqn_N, n_cos = n_cos, layers = iqn_layers).to(device)

            self.Q_tar.load_state_dict(self.Q.state_dict())

            self.eval_params = list(self.VDN.parameters()) + \
                               list(self.Q.parameters()) + \
                               list(self.node_representation_ship_feature.parameters()) + \
                               list(self.node_representation.parameters()) + \
                                list(self.node_representation_enemy.parameters())+\
                               list(self.func_missile_obs.parameters())+\
                               list(self.func_enemy_obs.parameters())

        self.optimizer = optim.Adam(self.eval_params, lr=learning_rate)
        from cfg import get_cfg
        cfg = get_cfg()
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_ratio)
        self.time_check = [[], []]
    def save_model(self, e, t, epsilon, path):
        torch.save({
            'e': e,
            't': t,
            'epsilon': epsilon,
            'Q': self.Q.state_dict(),
            'Q_tar': self.Q_tar.state_dict(),
            'node_representation_ship_feature': self.node_representation_ship_feature.state_dict(),
            'self.node_representation_enemy':self.node_representation_enemy.state_dict(),
            'node_representation': self.node_representation.state_dict(),
            'func_missile_obs': self.func_missile_obs.state_dict(),
            'func_enemy_obs': self.func_enemy_obs.state_dict(),
            'VDN': self.VDN.state_dict(),
            'VDN_target': self.VDN_target.state_dict(),
            'optimizer': self.optimizer.state_dict()}, "{}".format(path))

    def eval_check(self, eval):
        if eval == True:
            self.node_representation_ship_feature.eval()
            self.node_representation_enemy.eval()
            self.node_representation.eval()
            self.func_missile_obs.eval()
            self.func_enemy_obs.eval()
            self.Q.eval()
            self.Q_tar.eval()
        else:
            self.node_representation_ship_feature.train()
            self.node_representation_enemy.train()
            self.node_representation.train()
            self.func_missile_obs.train()
            self.func_enemy_obs.train()
            self.Q.train()
            self.Q_tar.train()

    #node_feature_enemy, edge_index_enemy, node_feature_enemy_next, edge_index_enemy_next
    def load_model(self, path):

        checkpoint = torch.load(path)
        e = checkpoint["e"]
        t = checkpoint["t"]
        epsilon = checkpoint["epsilon"]
        self.Q.load_state_dict(checkpoint["Q"])
        self.Q_tar.load_state_dict(checkpoint["Q_tar"])
        self.node_representation_ship_feature.load_state_dict(checkpoint["node_representation_ship_feature"])
        self.node_representation_enemy.load_state_dict(checkpoint["self.node_representation_enemy"])
        self.node_representation.load_state_dict(checkpoint["node_representation"])
        self.func_missile_obs.load_state_dict(checkpoint["func_missile_obs"])
        self.func_enemy_obs.load_state_dict(checkpoint["func_enemy_obs"])
        self.VDN.load_state_dict(checkpoint["VDN"])
        self.VDN_target.load_state_dict(checkpoint["VDN_target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # self.Q.head.remove_noise()
        # self.Q.head_y.remove_noise()
        # print("푸릇 셀러드?")
        #
        # for layer in self.Q.v_layer:
        #     if type(layer) is NoisyLinear:
        #         print("??? 되는겨?")
        #         layer.remove_noise()
        #
        # for layer in self.Q.advantage_layer:
        #     if type(layer) is NoisyLinear:
        #         print("??? 된겨?")
        #         layer.remove_noise()

        self.eval_params = list(self.VDN.parameters()) + \
                           list(self.Q.parameters()) + \
                           list(self.node_representation_ship_feature.parameters()) + \
                           list(self.node_representation.parameters()) + \
                           list(self.node_representation_enemy.parameters()) + \
                           list(self.func_missile_obs.parameters()) + \
                           list(self.func_enemy_obs.parameters())
        return e, t, epsilon

    def get_node_representation(self, missile_node_feature, ship_features, edge_index_missile,
                                n_node_features_missile,
                                enemy_node_feature,
                                enemy_edge_index,
                                n_node_features_enemy,
                                mini_batch=False):
        if self.GNN == 'GAT':
            if mini_batch == False:
                with torch.no_grad():


                    enemy_node_features = torch.tensor(enemy_node_feature, dtype=torch.float, device=device)
                    node_embedding_enemy = self.node_representation_enemy(enemy_node_features)
                    enemy_edge_index = torch.tensor(enemy_edge_index, dtype = torch.long, device = device)
                    node_embedding_enemy = self.func_enemy_obs(node_embedding_enemy,enemy_edge_index,  n_node_features_enemy, mini_batch = mini_batch)



                    ship_features = torch.tensor(ship_features, dtype=torch.float, device=device)
                    node_embedding_ship_features = self.node_representation_ship_feature(ship_features)


                    missile_node_feature = torch.tensor(missile_node_feature,dtype=torch.float,device=device).clone().detach()
                    node_embedding_missile_node = self.node_representation(missile_node_feature, missile=True)
                    edge_index_missile = torch.tensor(edge_index_missile, dtype=torch.long, device=device)
                    node_representation = self.func_missile_obs(node_embedding_missile_node, edge_index_missile,
                                                                 n_node_features_missile, mini_batch=mini_batch)

                    node_representation = torch.cat([node_embedding_ship_features, node_representation[0].unsqueeze(0), node_embedding_enemy[0].unsqueeze(0)], dim=1)
            else:

                enemy_node_features = torch.tensor(enemy_node_feature, dtype=torch.float, device=device)

                empty0 = list()
                for i in range(n_node_features_enemy):
                    node_embedding_enemy = self.node_representation_enemy(enemy_node_features[:, i, :], missile = True)
                    empty0.append(node_embedding_enemy)

                node_embedding_enemy = torch.stack(empty0)
                node_embedding_enemy = torch.einsum('ijk->jik', node_embedding_enemy)
                enemy_edge_index = torch.stack(enemy_edge_index)



                node_embedding_enemy = self.func_enemy_obs(node_embedding_enemy, enemy_edge_index,
                                                           n_node_features_enemy, mini_batch=mini_batch)



                ship_features = torch.tensor(ship_features,dtype=torch.float).to(device).squeeze(1)
                node_embedding_ship_features = self.node_representation_ship_feature(ship_features)

                missile_node_feature = torch.tensor(missile_node_feature, dtype=torch.float).to(device)
                empty = list()
                for i in range(n_node_features_missile):
                    node_embedding_missile_node = self.node_representation(missile_node_feature[:, i, :], missile=True)
                    empty.append(node_embedding_missile_node)
                node_embedding_missile_node = torch.stack(empty)
                node_embedding_missile_node = torch.einsum('ijk->jik', node_embedding_missile_node)
                edge_index_missile = torch.stack(edge_index_missile)
                node_representation = self.func_missile_obs(node_embedding_missile_node, edge_index_missile,
                                                             n_node_features_missile, mini_batch=mini_batch)

                node_representation = torch.cat([node_embedding_ship_features, node_representation[:, 0, :], node_embedding_enemy[:, 0, :]], dim=1)

        return node_representation

    def get_heterogeneous_adjacency_matrix(self, edge_index_enemy, edge_index_ally):
        A = []
        edge_index_enemy_transpose = deepcopy(edge_index_enemy)
        edge_index_enemy_transpose[1] = edge_index_enemy[0]
        edge_index_enemy_transpose[0] = edge_index_enemy[1]
        edge_index_ally_transpose = deepcopy(edge_index_ally)
        edge_index_ally_transpose[1] = edge_index_ally[0]
        edge_index_ally_transpose[0] = edge_index_ally[1]
        edges = [edge_index_enemy,
                 edge_index_enemy_transpose,
                 edge_index_ally,
                 edge_index_ally_transpose]
        for i, edge in enumerate(edges):
            edge = torch.tensor(edge, dtype=torch.long, device=device)
            value = torch.ones(edge.shape[1], dtype=torch.float, device=device)

            deg_inv_sqrt, deg_row, deg_col = _norm(edge.detach(),
                                                   self.num_nodes,
                                                   value.detach())  # row의 의미는 차원이 1이상인 node들의 index를 의미함

            value = deg_inv_sqrt[
                        deg_row] * value  # degree_matrix의 inverse 중에서 row에 해당되는(즉, node의 차원이 1이상인) node들만 골라서 value_tmp를 곱한다
            A.append((edge, value))

        edge = torch.stack((torch.arange(0, self.num_nodes), torch.arange(0, self.num_nodes))).type(
            torch.cuda.LongTensor)
        value = torch.ones(self.num_nodes).type(torch.cuda.FloatTensor)
        A.append((edge, value))
        return A

    def cal_Q(self, obs, actions, avail_actions_next, agent_id, cos, vdn, target=False):
        """
        node_representation
        - training 시        : batch_size X num_nodes X feature_size
        - action sampling 시 : num_nodes X feature_size
        """
        if vdn == True:
            if target == False:
                obs_n = obs
                q = self.Q(obs_n, cos, mini_batch=True)
                actions = torch.tensor(actions, device=device).long()
                act_n = actions[:, agent_id].unsqueeze(1)  # action.shape : (batch_size, 1)
                q = torch.gather(q, 1, act_n).squeeze(1)  # q.shape :      (batch_size, 1)
                return q
            else:
                with torch.no_grad():
                    obs_next = obs
                    obs_next = obs_next
                    avail_actions_next = torch.tensor(avail_actions_next, device=device).bool()
                    mask = avail_actions_next[:, agent_id]

                    q = self.Q(obs_next, cos, mini_batch=True)
                    q = q.masked_fill(mask == 0, float('-inf'))
                    act_n = torch.max(q, dim=1)[1].unsqueeze(1)

                    q_tar = self.Q_tar(obs_next, cos, mini_batch=True)
                    q_tar_max = torch.gather(q_tar, 1, act_n).squeeze(1)  # q.shape :      (batch_size, 1)

                    return q_tar_max
        else:
            if target == False:
                q = self.Q(obs, cos, mini_batch=True)
                actions = actions.long().to(device)
                q = torch.gather(q, 1, actions)
                return q.squeeze(1)
            else:
                obs_next = obs
                q_tar = self.Q(obs_next, cos, mini_batch=True)
                avail_actions_next = avail_actions_next.bool().to(device)
                mask = avail_actions_next
                q_tar = q_tar.masked_fill(mask == 0, float('-inf'))
                q_tar_max = torch.max(q_tar, dim=1)[0]
                return q_tar_max

    @torch.no_grad()
    def sample_action(self, node_representation, avail_action, epsilon):
        """
        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """
        mask = torch.tensor(avail_action, device=device).bool()
        action = []
        utility = list()
        cos, taus = self.Q.calc_cos(1)

        for n in range(self.num_agent):
            obs = node_representation[n]
            obs = obs.unsqueeze(0)
            Q = self.Q(obs, cos, mini_batch=False)
            Q = Q.masked_fill(mask[n, :] == 0, float('-inf'))
            greedy_u = torch.argmax(Q)

            if cfg.epsilon_greedy == True:

                if np.random.uniform(0, 1) >= epsilon:
                    u = greedy_u
                    #utility.append(Q[0][u].detach().item())
                    action.append(u)
                else:
                    mask_n = np.array(avail_action[n], dtype=np.float64)
                    u = np.random.choice(self.action_space, p=mask_n / np.sum(mask_n))
                   # utility.append(Q[0][u].detach().item())
                    action.append(u)
            else:

                u = greedy_u
               # utility.append(Q[0][u].detach().item())
                action.append(u)

        return action

    def learn(self, regularizer, vdn = False):

        node_features_missile, \
        ship_features, edge_indices_missile, \
        actions, \
        rewards, \
        dones, \
        node_features_missile_next, \
        ship_features_next, \
        edge_indices_missile_next, \
        avail_actions_next, status, status_next,priority,batch_index, node_feature_enemy, edge_index_enemy, node_feature_enemy_next, edge_index_enemy_next = self.buffer.sample(vdn = vdn)
        weight = (len(self.buffer.buffer[10])*torch.tensor(priority, dtype=torch.float, device = device))**(-self.beta)
        weight /= weight.max()
        # print("전", priority)
        # print("후", weight.detach().tolist())


        """
        node_features : batch_size x num_nodes x feature_size
        actions : batch_size x num_agents
        action_feature :     batch_size x action_size x action_feature_size
        avail_actions_next : batch_size x num_agents x action_size 
        """

        n_node_features_missile = self.n_node_feature_missile
        n_node_features_enemy = self.n_node_feature_enemy
        obs = self.get_node_representation(
            node_features_missile,
            ship_features,
            edge_indices_missile,
            n_node_features_missile,
            node_feature_enemy,
            edge_index_enemy,
            n_node_features_enemy,
            mini_batch=True)

        obs_next = self.get_node_representation(
            node_features_missile_next,
            ship_features_next,
            edge_indices_missile_next,
            n_node_features_missile,
            node_feature_enemy_next,
            edge_index_enemy_next,
            n_node_features_enemy,
            mini_batch=True)

        if vdn == True:
            dones = torch.tensor(dones, device=device, dtype=torch.float)
            rewards = torch.tensor(rewards, device=device, dtype=torch.float)
            cos, taus = self.Q.calc_cos(self.batch_size)



            q = [self.cal_Q(obs=obs,
                            actions=actions,
                            avail_actions_next=None,
                            agent_id=agent_id,
                            target=False,
                            cos=cos,vdn = vdn) for agent_id in range(self.num_agent)]
            q_tot = torch.stack(q, dim=1)
            q_tar = [self.cal_Q(obs=obs_next,
                                actions=None,
                                avail_actions_next=avail_actions_next,
                                agent_id=agent_id,
                                target=True,
                                cos=cos, vdn = vdn) for agent_id in range(self.num_agent)]

            q_tot_tar = torch.stack(q_tar, dim=1)

            q_tot = q_tot          # status/status.sum(dim = 1, keepdims = True)
            q_tot_tar = q_tot_tar  # status_next/status_next.sum(dim = 1, keepdims = True)
            q_tot     = self.VDN(q_tot)
            q_tot_tar = self.VDN_target(q_tot_tar)
            rewards_1_step = rewards[:, 0].unsqueeze(1)

            rewards_k_step = rewards[:, 1:]
            masked_n_step_bootstrapping = dones*torch.cat([rewards_k_step, q_tot_tar.unsqueeze(1)], dim = 1)
            discounted_n_step_bootstrapping = self.gamma_n_step*torch.cat([rewards_1_step, masked_n_step_bootstrapping], dim = 1)
            td_target = discounted_n_step_bootstrapping.sum(dim=1)

            delta = (td_target - q_tot).detach().tolist()
            self.buffer.update_transition_priority(batch_index = batch_index, delta = delta)

            loss = F.huber_loss(weight*q_tot, weight*td_target.detach())#
            self.optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.eval_params, 1)
            self.optimizer.step()
            self.scheduler.step()
            tau = 1e-3
            for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
                target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        else:
            sampled_indexes = [np.random.choice(self.agent_index, p = np.array(sta)/np.sum(sta)) for sta in status]
            obs = torch.stack([obs[b, sampled_indexes[b], :] for b in range(self.batch_size)])
            obs_next = torch.stack([obs_next[b, sampled_indexes[b], :] for b in range(self.batch_size)])
            rewards = torch.gather(torch.tensor(rewards), 1, torch.tensor(sampled_indexes).long().unsqueeze(1)).to(device)
            actions = torch.gather(torch.tensor(actions), 1, torch.tensor(sampled_indexes).long().unsqueeze(1))
            avail_actions_next = torch.tensor([avail_actions_next[b][sampled_indexes[b]] for b in range(self.batch_size)])
            cos, taus = self.Q.calc_cos(self.batch_size)


            q_tot =self.cal_Q(obs=obs,
                       actions=actions,
                       avail_actions_next=None,
                       agent_id=None,
                       target=False,
                       vdn = False,
                       cos=cos)

            q_tot_tar =self.cal_Q(obs=obs_next,
                       actions=None,
                       avail_actions_next=avail_actions_next,
                       agent_id=None,
                       target=True,
                       vdn=False,
                       cos=cos)
            """
            q_tot.shape : batch_size, num_agent
            rewards.shape : batch_size, num_agent
            """
            dones = torch.tensor(dones, device=device, dtype=torch.float)
            td_target = rewards.squeeze(1) + self.gamma * (1-dones) * q_tot_tar
            loss1 = F.huber_loss(q_tot, td_target.detach())
            loss = loss1
            self.optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.eval_params, 1)
            self.optimizer.step()
            tau = 1e-3
            for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
                target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
            if cfg.epsilon_greedy == False:
                self.Q.reset_noise_net()
                self.Q_tar.reset_noise_net()

        return loss