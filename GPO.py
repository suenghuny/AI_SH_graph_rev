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

class Network(nn.Module):
    def __init__(self, state_size, state_action_size, layers=[8,12]):
        super(Network, self).__init__()
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
        self.network = Network(state_size = n_representation_ship + n_representation_missile + 2,
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



        self.optimizer1 = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.optimizer2 = optim.Adam(self.network.parameters(), lr=learning_rate_critic)

        #self.scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=self.learning_rate, total_steps=30000)
    def get_node_representation(self, missile_node_feature, ship_features, edge_index_missile,n_node_features_missile,mini_batch=False):
        #print("??????????????")
        if mini_batch == False:
            with torch.no_grad():
                print("????????????????????")
                ship_features = torch.tensor(ship_features, dtype=torch.float, device=device)
                print(ship_features.shape, "skskskds")
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
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, mask_lst, done_lst = [], [], [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a, mask, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            mask_lst.append(mask)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        s, a, r, s_prime, prob_a, mask, done = torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(a_lst).to(device), \
                                               torch.tensor(r_lst, dtype=torch.float).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                                               torch.tensor(prob_a_lst).to(device), torch.tensor(mask_lst).to(device), torch.tensor(done_lst, dtype=torch.float).to(device)

        self.data = []

        return s, a, r, s_prime, prob_a, mask, done

    def get_action(self, s, possible_actions= [True, True]):
        self.network.eval()
        s = torch.tensor(s).to(device).unsqueeze(0)
        s = s.expand([self.action_size, self.state_size])
        obs_n_action = torch.concat([s, self.action_encoding], dim = 1)
        logit = self.network.pi(obs_n_action).squeeze(1)
        mask = torch.tensor(possible_actions, device=device).bool()
        logit = logit.masked_fill(mask == 0, - 1e8)
        #print("전", logit)
        prob = torch.softmax(logit, dim=-1)
        #print("후", prob)

        m = Categorical(prob)
        a = m.sample().item()
        return a, prob, mask

    def train(self):
        self.network.train()
        s, a, r, s_prime, prob_a, mask, done = self.make_batch()
        avg_loss = 0.0

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.network.v(s_prime) * done
            delta = td_target - self.network.v(s)
            delta = delta.cpu().detach().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)






            s_expand = s.unsqueeze(1).expand([s.shape[0], self.action_size, self.state_size])
            a_expand = self.action_encoding.unsqueeze(0).expand([s.shape[0], self.action_size, self.action_size])
            obs_n_act = torch.concat([s_expand, a_expand], dim = 2)

            logit = torch.stack([self.network.pi(obs_n_act[:, i]) for i in range(self.action_size)])
            logit = torch.einsum('ijk->jik', logit).squeeze(2)
            logit = logit.masked_fill(mask == 0, -1e8)
            pi = torch.softmax(logit, dim=-1)
            #print("후", pi)
            #print(pi)
            action_indices = a.nonzero(as_tuple= True)[1].long().unsqueeze(1)


            pi_a = pi.gather(1, action_indices)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss1 = -torch.min(surr1, surr2)
            loss2 = 0.2 * F.smooth_l1_loss(self.network.v(s), td_target.detach())
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss1.mean().backward()
            loss2.mean().backward()
            self.optimizer1.step()
            self.optimizer2.step()
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
