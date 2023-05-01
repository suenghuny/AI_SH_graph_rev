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
    def __init__(self, state_size, action_size, batch_size, layer_size=64, N=12, layers = [72, 64, 56, 48, 32]):
        super(IQN, self).__init__()
        self.input_shape = state_size
        self.batch_size = batch_size
        # print(state_size)
        self.action_size = action_size
        self.K = 32
        self.N = N
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(device)
        self.head = NoisyLinear(self.input_shape, layer_size)  # cound be a cnn
        self.head_y = NoisyLinear(self.input_shape, layer_size)  # cound be a cnn
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)



        self.noisylinears_for_advantage = OrderedDict()
        last_layer = layer_size
        for i in range(len(layers)):
            layer = layers[i]
            if i != len(layers)-1:
                self.noisylinears_for_advantage['linear{}'.format(i)]= NoisyLinear(last_layer, layer)
                self.noisylinears_for_advantage['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.noisylinears_for_advantage['activation{}'.format(i)] = nn.ReLU()
                last_layer = layer
            else:
                self.noisylinears_for_advantage['linear{}'.format(i)] = NoisyLinear(last_layer, self.action_size)

        self.noisylinears_for_v = OrderedDict()
        last_layer = layer_size
        for i in range(len(layers)):
            layer = layers[i]
            if i != len(layers) - 1:
                self.noisylinears_for_v['linear{}'.format(i)]= NoisyLinear(last_layer, layer)
                self.noisylinears_for_v['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.noisylinears_for_v['activation{}'.format(i)] = nn.ReLU()
                last_layer = layer
            else:
                self.noisylinears_for_v['linear{}'.format(i)] = NoisyLinear(last_layer, 1)

        self.advantage_layer = nn.Sequential(self.noisylinears_for_advantage)
        self.v_layer = nn.Sequential(self.noisylinears_for_v)
        self.advantage_layer.apply(weight_init_xavier_uniform)
        self.v_layer.apply(weight_init_xavier_uniform)
        # for layer in self.v_layer:
        #     print(layer)
        self.reset_noise_net()


    def reset_noise_net(self):
        for layer in self.v_layer:
            if type(layer) is NoisyLinear:
                layer.sample_noise()

        for layer in self.advantage_layer:
            if type(layer) is NoisyLinear:
                layer.sample_noise()




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
        out = self.advantage_layer(x)
        quantiles = out.view(batch_size, N, self.action_size)
        a = quantiles.mean(dim=1)
        y = torch.relu(self.head_y(input.to(device)))  # x의 shape는 batch_size, layer_size
        cos_y = cos_x
        y = (y.unsqueeze(1) * cos_y).view(batch_size * N, self.layer_size)  # 이부분이 phsi * phi에 해당하는 부분
        out = self.v_layer(y)
        quantiles = out.view(batch_size, N, 1)
        v = quantiles.mean(dim=1)
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
            if i != len(layers)-1:
                self.linears['linear{}'.format(i)]= nn.Linear(last_layer, layer)
                self.linears['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.linears['activation{}'.format(i)] = nn.ReLU()
                last_layer = layer
            else:
                self.linears['linear{}'.format(i)] = NoisyLinear(last_layer, n_representation_obs)

        self.node_embedding = nn.Sequential(self.linears)
        self.node_embedding.apply(weight_init_xavier_uniform)


    def forward(self, node_feature, machine=False):
        node_representation = self.node_embedding(node_feature)
        return node_representation


class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, num_agent, action_size, n_step):
        self.buffer = deque()

        self.step_count_list = list()
        for _ in range(11):
            self.buffer.append(deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.num_agent = num_agent
        self.agent_id = np.eye(self.num_agent).tolist()
        self.one_hot_actions = np.eye(action_size).tolist()
        self.batch_size = batch_size
        self.step_count = 0
        self.rewards_store = list()
        self.n_step = n_step

    def pop(self):
        self.buffer.pop()

    def memory(self, node_feature_machine, num_waiting_operations, edge_index_machine, action, reward, done,
               avail_action, status):


        self.buffer[1].append(node_feature_machine)
        #print(self.buffer[1])
        self.buffer[2].append(num_waiting_operations)

        self.buffer[3].append(edge_index_machine)
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

        if self.step_count < self.buffer_size:
            self.step_count_list.append(self.step_count)
            self.step_count += 1

        #print(len(self.buffer[7]), len(self.buffer[10]), len(self.step_count_list))


    def generating_mini_batch(self, datas, batch_idx, cat):
        for s in batch_idx:
            if cat == 'node_feature_machine':
                yield datas[1][s]
            if cat == 'num_waiting_operations':
                yield datas[2][s]
            if cat == 'edge_index_machine':
                yield torch.sparse_coo_tensor(datas[3][s],
                                              torch.ones(torch.tensor(datas[3][s]).shape[1]),
                                              (self.num_agent, self.num_agent)).to_dense()
            if cat == 'action':
                yield datas[4][s]
            if cat == 'reward':
                yield datas[5][s]
            if cat == 'done':
                yield datas[6][s]
            if cat == 'node_feature_machine_next':
                yield datas[1][s + self.n_step]
            if cat == 'num_waiting_operations_next':
                yield datas[2][s + self.n_step]
            if cat == 'edge_index_machine_next':
                yield torch.sparse_coo_tensor(datas[3][s + 1],
                                              torch.ones(torch.tensor(datas[3][s + 1]).shape[1]),
                                              (self.num_agent, self.num_agent)).to_dense()
            if cat == 'avail_action_next':
                yield datas[7][s + self.n_step]
            if cat == 'status':
                yield datas[8][s]
            if cat == 'status_next':
                yield datas[8][s+ self.n_step]

            if cat == 'priority':
                yield datas[10][s]

    #def search_sample_space(self, sampled_batch_idx):

    def update_transition_priority(self, batch_index, delta):
        #print(list(self.buffer[10]))
        copied_delta_store = deepcopy(list(self.buffer[10]))
        delta = np.abs(delta) + np.min(copied_delta_store)
        priority = np.array(copied_delta_store).astype(float)
        batch_index = batch_index.astype(int)
        priority[batch_index] = delta
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
            p = (np.array(priority)/np.array(priority).sum()).tolist()
            #print(p)
            sampled_batch_idx = np.random.choice(step_count_list[:-self.n_step+1], size=self.batch_size, p = p)

        node_feature_machine = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_machine')
        node_features_machine = list(node_feature_machine)

        action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action')
        actions = list(action)

        num_waiting_operations = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='num_waiting_operations')
        num_waiting_operations = list(num_waiting_operations)

        num_waiting_operations_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                                 cat='num_waiting_operations_next')
        num_waiting_operations_next = list(num_waiting_operations_next)

        edge_index_machine = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_machine')

        edge_indices_machine = list(edge_index_machine)

        reward = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='reward')
        rewards = list(reward)

        done = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='done')
        dones = list(done)

        #
        # node_feature_job_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_job_next')
        # node_features_job_next = list(node_feature_job_next)

        node_feature_machine_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                               cat='node_feature_machine_next')
        node_features_machine_next = list(node_feature_machine_next)

        # edge_index_job_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_job_next')
        # edge_indices_job_next = list(edge_index_job_next)

        edge_index_machine_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                             cat='edge_index_machine_next')
        edge_indices_machine_next = list(edge_index_machine_next)

        avail_action_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action_next')
        avail_actions_next = list(avail_action_next)

        status = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='status')
        status = list(status)


        status_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='status_next')
        status_next = list(status_next)

        priority = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='priority')
        priority = list(priority)



        return node_features_machine, num_waiting_operations, edge_indices_machine, actions, rewards, dones, node_features_machine_next, num_waiting_operations_next, edge_indices_machine_next, avail_actions_next, status, status_next,priority, sampled_batch_idx


# iqn_layers = cfg.iqn_layers,
# node_embedding_layers_job = cfg.job_layers,
# node_embedding_layers_machine = cfg.machine_layers,
class Agent:
    def __init__(self,
                 num_agent,
                 feature_size_job,
                 feature_size_machine,

                 iqn_layers,
                 node_embedding_layers_job,
                 node_embedding_layers_machine,
                 n_multi_head,
                 n_representation_job,
                 n_representation_machine,

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
                 n_node_feature,
                 n_step,
                 beta):
        self.n_step = n_step
        self.num_agent = num_agent
        self.feature_size_job = feature_size_job
        self.feature_size_machine = feature_size_machine
        # self.hidden_size_meta_path = hidden_size_meta_path
        # self.hidden_size_obs = hidden_size_obs
        # self.hidden_size_comm = hidden_size_comm
        self.n_multi_head = n_multi_head
        self.teleport_probability = teleport_probability
        self.beta = beta

        self.n_representation_job = n_representation_job
        self.n_representation_machine = n_representation_machine

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
        self.buffer = Replay_Buffer(self.buffer_size, self.batch_size, n_node_feature, self.action_size, n_step = self.n_step
                                    )
        self.n_node_feature = n_node_feature
        self.action_space = [i for i in range(self.action_size)]

        self.GNN = GNN

        self.gamma_n_step = torch.tensor([[self.gamma**i for i in range(self.n_step+1)] for _ in range(self.batch_size)], dtype = torch.float, device = device)


        if self.GNN == 'GAT':

            self.node_representation_job_obs = NodeEmbedding(feature_size=feature_size_job,
                                                             n_representation_obs=n_representation_job,
                                                             layers = node_embedding_layers_job).to(device)  # 수정사항

            self.node_representation = NodeEmbedding(feature_size=feature_size_machine,
                                                     n_representation_obs=n_representation_machine,
                                                     layers = node_embedding_layers_machine).to(device)  # 수정사항


            self.func_machine_comm = GAT(nfeat=n_representation_machine,
                                         nhid=hidden_size_comm,
                                         nheads=n_multi_head,
                                         nclass=n_representation_machine,
                                         dropout=dropout,
                                         alpha=0.2,
                                         mode='observation',
                                         batch_size=self.batch_size,
                                         teleport_probability=self.teleport_probability).to(device)  # 수정사항

            self.Q = IQN(n_representation_job + n_representation_machine, self.action_size,
                         batch_size=self.batch_size).to(device)
            self.Q_tar = IQN(n_representation_job + n_representation_machine, self.action_size,
                             batch_size=self.batch_size).to(device)
            # Network(n_representation_job+n_representation_machine+5, hidden_size_Q).to(device)
            # self.Q_tar = Network(n_representation_job+n_representation_machine+5, hidden_size_Q, self.action_size).to(device)
            self.Q_tar.load_state_dict(self.Q.state_dict())

            self.eval_params = list(self.VDN.parameters()) + \
                               list(self.Q.parameters()) + \
                               list(self.node_representation_job_obs.parameters()) + \
                               list(self.node_representation.parameters()) + \
                               list(self.func_machine_comm.parameters())

        self.optimizer = optim.Adam(self.eval_params, lr=learning_rate)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=38000, gamma=0.1)
        self.time_check = [[], []]

    def save_model(self, e, t, epsilon, path):
        torch.save({
            'e': e,
            't': t,
            'epsilon': epsilon,
            'Q': self.Q.state_dict(),
            'Q_tar': self.Q_tar.state_dict(),
            'node_representation_job_obs': self.node_representation_job_obs.state_dict(),
            'node_representation': self.node_representation.state_dict(),
            'func_machine_comm': self.func_machine_comm.state_dict(),
            'VDN': self.VDN.state_dict(),
            'VDN_target': self.VDN_target.state_dict(),
            'optimizer': self.optimizer.state_dict()}, "{}".format(path))

    def eval_check(self, eval):
        if eval == True:
            self.node_representation_job_obs.eval()
            self.node_representation.eval()
            self.func_machine_comm.eval()
            self.Q.eval()
            self.Q_tar.eval()
        else:
            self.node_representation_job_obs.train()
            self.node_representation.train()
            self.func_machine_comm.train()
            self.Q.train()
            self.Q_tar.train()

    def load_model(self, path):
        checkpoint = torch.load(path)
        e = checkpoint["e"]
        t = checkpoint["t"]
        epsilon = checkpoint["epsilon"]
        self.Q.load_state_dict(checkpoint["Q"])
        self.Q_tar.load_state_dict(checkpoint["Q_tar"])
        self.node_representation_job_obs.load_state_dict(checkpoint["node_representation_job_obs"])
        self.node_representation.load_state_dict(checkpoint["node_representation"])
        self.func_machine_comm.load_state_dict(checkpoint["func_machine_comm"])
        self.VDN.load_state_dict(checkpoint["VDN"])
        self.VDN_target.load_state_dict(checkpoint["VDN_target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.eval_params = list(self.VDN.parameters()) + \
                           list(self.Q.parameters()) + \
                           list(self.node_representation_job_obs.parameters()) + \
                           list(self.node_representation.parameters()) + \
                           list(self.func_machine_comm.parameters())
        return e, t, epsilon

    def get_node_representation(self, node_feature_machine, num_waiting_operations, edge_index_machine,
                                n_node_features_machine, mini_batch=False):
        if self.GNN == 'GAT':
            if mini_batch == False:
                with torch.no_grad():
                    num_waiting_operations = torch.tensor(num_waiting_operations, dtype=torch.float, device=device)
                    node_embedding_num_waiting_operation = self.node_representation_job_obs(num_waiting_operations)

                    node_feature_machine = torch.tensor(node_feature_machine,dtype=torch.float,device=device).clone().detach()
                    node_embedding_machine_obs = self.node_representation(node_feature_machine, machine=True)
                    edge_index_machine = torch.tensor(edge_index_machine, dtype=torch.long, device=device)
                    node_representation = self.func_machine_comm(node_embedding_machine_obs, edge_index_machine,
                                                                 n_node_features_machine, mini_batch=mini_batch)

                    node_representation = torch.cat([node_embedding_num_waiting_operation, node_representation[0].unsqueeze(0)], dim=1)
            else:
                node_feature_machine = torch.tensor(node_feature_machine, dtype=torch.float).to(device)
                num_waiting_operations = torch.tensor(num_waiting_operations,dtype=torch.float).to(device).squeeze(1)

                node_embedding_num_waiting_operation = self.node_representation_job_obs(num_waiting_operations)

                empty = list()
                for i in range(n_node_features_machine):
                    node_embedding_machine_obs = self.node_representation(node_feature_machine[:, i, :], machine=True)
                    empty.append(node_embedding_machine_obs)
                node_embedding_machine_obs = torch.stack(empty)
                node_embedding_machine_obs = torch.einsum('ijk->jik', node_embedding_machine_obs)

                edge_index_machine = torch.stack(edge_index_machine)

                node_representation = self.func_machine_comm(node_embedding_machine_obs, edge_index_machine,
                                                             n_node_features_machine, mini_batch=mini_batch)

                node_representation = torch.cat(
                    [node_embedding_num_waiting_operation, node_representation[:, 0, :]], dim=1)

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

                    q = self.Q_tar(obs_next, cos, mini_batch=True)

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
            u = greedy_u.detach().item()
            utility.append(Q[0][u].detach().item())
            action.append(u)

        return action

    def learn(self, regularizer, vdn = False):

        # import time
        # start = time.time()
        node_features_machine, num_waiting_operations, edge_indices_machine, actions, rewards, dones, node_features_machine_next, num_waiting_operations_next, edge_indices_machine_next, avail_actions_next, status, status_next,priority,batch_index = self.buffer.sample(vdn = vdn)
        weight = (len(self.buffer.buffer[10])*torch.tensor(priority, dtype=torch.float, device = device))**(-self.beta)
        weight /= weight.max()

        """
        node_features : batch_size x num_nodes x feature_size
        actions : batch_size x num_agents
        action_feature :     batch_size x action_size x action_feature_size
        avail_actions_next : batch_size x num_agents x action_size 
        """
        n_node_features_machine = torch.tensor(node_features_machine).shape[1]
        obs = self.get_node_representation(
            node_features_machine,
            num_waiting_operations,
            edge_indices_machine,
            n_node_features_machine,
            mini_batch=True)

        obs_next = self.get_node_representation(
            node_features_machine_next,
            num_waiting_operations_next,
            edge_indices_machine_next,
            n_node_features_machine,
            mini_batch=True)

        if vdn == True:
            #print(avail_actions_next)

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

            q_tot = q_tot #* status/status.sum(dim = 1, keepdims = True)
            q_tot_tar = q_tot_tar#* status_next/status_next.sum(dim = 1, keepdims = True)
            q_tot = self.VDN(q_tot)
            q_tot_tar = self.VDN_target(q_tot_tar)
            rewards_1_step = rewards[:,0].unsqueeze(1)
            rewards_k_step = rewards[:, 1:]
            masked_n_step_bootstrapping = dones*torch.cat([rewards_k_step, q_tot_tar.unsqueeze(1)], dim = 1)
            discounted_n_step_bootstrapping = self.gamma_n_step*torch.cat([rewards_1_step, masked_n_step_bootstrapping], dim = 1)
            td_target = discounted_n_step_bootstrapping.sum(dim=1)
            delta = (td_target - q_tot).detach().tolist()
            self.buffer.update_transition_priority(batch_index = batch_index, delta = delta)
            loss1 = F.huber_loss(weight*q_tot, weight*td_target.detach())
            loss = loss1
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_params, 1)
            self.optimizer.step()
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

            self.Q.reset_noise_net()
            self.Q_tar.reset_noise_net()

        return loss