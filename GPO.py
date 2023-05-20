import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from GDN import NodeEmbedding
import torch
import torch.optim as optim
from GAT.model import GAT
from GAT.layers import device
from cfg import get_cfg
cfg = get_cfg()
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPONetwork(nn.Module):
    def __init__(self, state_size, state_action_size, layers=[8,12]):
        super(PPONetwork, self).__init__()
        self.state_size = state_size
        self.state_action_size = state_action_size
        self.NN_sequential = OrderedDict()

        self.fc_pi = nn.Linear(state_action_size, layers[0])
        self.bn_pi = nn.BatchNorm1d(layers[0])

        self.fc_v = nn.Linear(state_size, layers[0])
        self.bn_v = nn.BatchNorm1d(layers[0])
        self.fcn = OrderedDict()

        last_layer = layers[0]
        for i in range(1, len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.fcn['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.fcn['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.fcn['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            #else:
        self.forward_cal = nn.Sequential(self.fcn)
        self.output_pi = nn.Linear(last_layer, 1)
        self.output_v = nn.Linear(last_layer, 1)




    def pi(self, x):

        x = self.fc_pi(x)
        # print(x.shape)
        # print(x)
        #x = self.bn_pi(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        pi = self.output_pi(x)
        return pi

    def v(self, x):
        x = self.fc_v(x)
        #x = self.bn_v(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        v = self.output_v(x)
        return v





class Agent:
    def __init__(self,

                 action_size,
                 feature_size_ship,
                 feature_size_missile,
                 feature_size_action,

                 n_node_feature_missile,
                 n_representation_ship = cfg.n_representation_ship,
                 n_representation_missile = cfg.n_representation_missile,
                 n_representation_action = cfg.n_representation_action,


                 node_embedding_layers_action = list(eval(cfg.action_layers)),
                 node_embedding_layers_ship = list(eval(cfg.ship_layers)),
                 node_embedding_layers_missile = list(eval(cfg.missile_layers)),

                 hidden_size_comm = cfg.hidden_size_comm,
                 n_multi_head = cfg.n_multi_head,
                 dropout = 0.6,

                 learning_rate=cfg.lr,
                 learning_rate_critic=cfg.lr_critic,
                 gamma=cfg.gamma,
                 lmbda=cfg.lmbda,
                 eps_clip = cfg.eps_clip,
                 K_epoch = cfg.K_epoch,
                 layers=list(eval(cfg.ppo_layers))
                 ):


        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.data = []
        #print("확인", n_representation_ship, n_representation_missile, 2, n_representation_action)
        self.network = PPONetwork(state_size = n_representation_ship+n_representation_missile + 2,
                               state_action_size = n_representation_ship+n_representation_missile + 2 + n_representation_action,
                               layers = layers).to(device)


        self.node_representation_action_feature = NodeEmbedding(feature_size=feature_size_action,
                                                                n_representation_obs=n_representation_action,
                                                                layers=node_embedding_layers_action).to(device)  # 수정사항

        self.node_representation_ship_feature = NodeEmbedding(feature_size=feature_size_ship,
                                                              n_representation_obs=n_representation_ship,
                                                              layers=node_embedding_layers_ship).to(device)  # 수정사항

        self.node_representation = NodeEmbedding(feature_size=feature_size_missile,
                                                 n_representation_obs=n_representation_missile,
                                                 layers=node_embedding_layers_missile).to(device)  # 수정사항

        self.func_missile_obs = GAT(nfeat=n_representation_missile,
                                    nhid=hidden_size_comm,
                                    nheads=n_multi_head,
                                    nclass=n_representation_missile + 2,
                                    dropout=dropout,
                                    alpha=0.2,
                                    mode='communication',
                                    batch_size= 1,
                                    teleport_probability=cfg.teleport_probability).to(device)  # 수정사항

        self.eval_params = list(self.network.parameters()) + \
                           list(self.node_representation_action_feature.parameters()) + \
                           list(self.node_representation_ship_feature.parameters()) + \
                           list(self.node_representation.parameters()) + \
                           list(self.func_missile_obs.parameters())

        self.n_node_feature_missile = n_node_feature_missile


        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        #self.scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=self.learning_rate, total_steps=30000)
    def eval_check(self, eval):
        if eval == True:
            self.network.eval()
            self.node_representation_action_feature.eval()
            self.node_representation_ship_feature.eval()
            self.node_representation.eval()
            self.func_missile_obs.eval()
        else:
            self.network.train()
            self.node_representation_action_feature.train()
            self.node_representation_ship_feature.train()
            self.node_representation.train()
            self.func_missile_obs.train()

    def get_node_representation(self, missile_node_feature, ship_features, edge_index_missile,n_node_features_missile,mini_batch=False):
        #print("??????????????")
        if mini_batch == False:
            with torch.no_grad():
                ship_features = torch.tensor(ship_features, dtype=torch.float, device=device)
                node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
                missile_node_feature = torch.tensor(missile_node_feature,dtype=torch.float,device=device).clone().detach()
                node_embedding_missile_node = self.node_representation(missile_node_feature, missile=True)
                edge_index_missile = torch.tensor(edge_index_missile, dtype=torch.long, device=device)
                node_representation = self.func_missile_obs(node_embedding_missile_node, edge_index_missile,
                                                             n_node_features_missile, mini_batch=mini_batch)
                node_representation = torch.cat([node_embedding_ship_features, node_representation[0].unsqueeze(0)], dim=1)
        else:
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
            node_representation = torch.cat([node_embedding_ship_features, node_representation[:, 0, :],  ], dim=1)
        return node_representation



    def put_data(self, transition):
        self.data.append(transition)


    def make_batch(self):

        ship_feature_list = list()
        edge_indices_list = list()
        missile_node_feature_list = list()
        action_feature_list = list()
        a_list = list()

        ship_feature_next_list = list()
        edge_indices_next_list = list()
        missile_node_feature_next_list = list()

        r_list = list()
        prob_list = list()
        done_list = list()
        avail_action_list = list()
        a_indices_list = list()

        for t in range(len(self.data)):
            ship_feature, \
            edge_index, \
            missile_node_feature, \
            action_feature, \
            a, \
            r, \
            prob,\
            mask, \
            done,\
            avail_actions,\
            a_index= self.data[t]
            avail_action_list.append(avail_actions)
            ship_feature_list.append(ship_feature)
            missile_node_feature_list.append(missile_node_feature)
            action_feature_list.append(action_feature)
            a_indices_list.append(a_index)

            a_list.append(a)
            r_list.append(r)
            prob_list.append(prob)
            done_list.append(done)

            if t == len(self.data)-1:
                ship_feature_next = [[0]*len(ship_feature[0])]
                edge_index_next = [[],[]]
                missile_node_feature_next = [[0]*len(missile_node_feature[0]) for _ in range(self.n_node_feature_missile)]
            else:
                ship_feature_next, \
                edge_index_next, \
                missile_node_feature_next, \
                _, \
                _, \
                _, \
                _, \
                _, \
                _ ,\
                _ , \
                _= self.data[t+1]

            ship_feature_next_list.append(ship_feature_next)
            missile_node_feature_next_list.append(missile_node_feature_next)

            edge_indices = torch.sparse_coo_tensor(edge_index,
                                    torch.ones(len(edge_index[0])),
                                    (self.n_node_feature_missile, self.n_node_feature_missile)).to_dense()


            edge_indices_next = torch.sparse_coo_tensor(edge_index_next,
                                    torch.ones(len(edge_index_next[0])),
                                    (self.n_node_feature_missile, self.n_node_feature_missile)).to_dense()



            edge_indices_list.append(edge_indices)
            edge_indices_next_list.append(edge_indices_next)


        ship_feature = torch.tensor(ship_feature_list).to(device).float().squeeze(1)
        missile_node_feature = torch.tensor(missile_node_feature_list).to(device).float()
        action_feature = torch.tensor(action_feature_list).to(device).float()
        a = torch.tensor(a_list).to(device).float()
        ship_feature_next = torch.tensor(ship_feature_next_list).to(device).float().squeeze(1)
        missile_node_feature_next = torch.tensor(missile_node_feature_next_list).to(device).float()

        edge_indices = edge_indices_list
        edge_indices_next = edge_indices_next_list

        r = torch.tensor(r_list).float()
        prob_a = torch.tensor(prob_list).float()
        done = torch.tensor(done_list).float()
        avail_action = torch.tensor(avail_action_list).bool()
        a_indices = torch.tensor(a_indices_list).long()

        self.data = []


        return ship_feature,missile_node_feature,action_feature,a,ship_feature_next,missile_node_feature_next,edge_indices,edge_indices_next,r,prob_a,done, avail_action, a_indices

    def sample_action(self, s, possible_actions, action_feature):
        dummy_action_feature = action_feature
        action_feature = torch.tensor(action_feature, dtype = torch.float).to(device)
        node_embedding_action = self.node_representation_action_feature(action_feature)

        s = s.expand([node_embedding_action.shape[0], s.shape[1]])
        obs_n_action = torch.cat([s, node_embedding_action], dim = 1)

        logit = [self.network.pi(obs_n_action[i].unsqueeze(0)) for i in range(obs_n_action.shape[0])]
        logit = torch.stack(logit).view(1, -1)
        mask = torch.tensor(possible_actions, device=device).bool()
        logit = logit.masked_fill(mask == 0, - 1e8)

        prob = torch.softmax(logit, dim=-1)
        #print("후", prob)

        m = Categorical(prob)
        a = m.sample().item()
        a_index = a
        prob_a = prob.squeeze(0)[a]

        action_blue = dummy_action_feature[a]

        return action_blue, prob_a, mask, a_index

    def learn(self):
        ship_features,missile_node_feature,action_feature,a,ship_features_next,missile_node_feature_next,edge_indices,edge_indices_next,r,prob_a,done,mask,a_indices = self.make_batch()
        avg_loss = 0.0

        a_indices = a_indices.unsqueeze(1)
        for i in range(self.K_epoch):
            s = self.get_node_representation(missile_node_feature,
                                             ship_features,
                                             edge_indices,
                                             self.n_node_feature_missile,
                                             mini_batch= True)
            s_prime = self.get_node_representation(missile_node_feature_next,
                                                   ship_features_next,
                                                   edge_indices_next,
                                                    self.n_node_feature_missile, mini_batch=True)


            td_target = r.unsqueeze(1) + self.gamma * self.network.v(s_prime) * (1-done).unsqueeze(1)
            v_s = self.network.v(s)
            #print(td_target.shape,  self.network.v(s).shape)
            delta = td_target - v_s
            delta = delta.cpu().detach().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)
            #print(action_feature.shape)
            action_embedding = [self.node_representation_action_feature(action_feature[:, i]) for i in range(action_feature.shape[1])]
            action_embedding = torch.stack(action_embedding)
            action_embedding = torch.einsum('ijk -> jik', action_embedding)

            s_expand = s.unsqueeze(1).expand([s.shape[0], action_feature.shape[1], s.shape[1]])
            #print(s_expand.shape, action_feature.shape)
            obs_n_act = torch.cat([s_expand, action_embedding], dim= 2)

            logit = torch.stack([self.network.pi(obs_n_act[:, i]) for i in range(action_feature.shape[1])])
            logit = torch.einsum('ijk->jik', logit).squeeze(2)
            mask = mask.squeeze(1)
            logit = logit.masked_fill(mask == 0, -1e8)

            pi = torch.softmax(logit, dim=-1)
            pi_a = pi.gather(1, a_indices)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2)+0.2 * F.smooth_l1_loss(v_s, td_target.detach())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            #avg_loss += loss.mean().item()

        return avg_loss / self.K_epoch

    def save_network(self, e, file_dir):
        torch.save({"episode": e,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode%d.pt" % e)



# action_encoding = np.eye(action_size, dtype = np.float)
#
# cumul = list()
# for n_epi in range(10000):
#     s = env.reset()
#     done = False
#     epi_reward = 0
#     s = s[0]
#     step =0
#     while not done:
#         a, prob, mask = agent.get_action(s)
#         s_prime, r, done, info, _ = env.step(a)
#         mask = [True, True]
#         epi_reward+= r
#         step+=1
#         agent.put_data((s, action_encoding[a], r, s_prime, prob[a].item(), mask, done))
#         s = s_prime
#     cumul.append(epi_reward)
#     n = 100
#     if n_epi > n:
#         print(np.mean(cumul[-n:]))
#     agent.train()
#     #print(r)
