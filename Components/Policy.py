import numpy as np
from utils import *
from cfg import get_cfg
cfg = get_cfg()
import torch
import random

np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Policy:

    def __init__(self, env, rule = 'rule2', temperatures = [10, 20]):
        self.env = env
        self.rule = rule
        self.temperature1 = temperatures[0]
        self.temperature2 = temperatures[1]


    def get_action(self, avail_action_list, target_distance_list, air_alert):
        if self.rule == 'rule1':
            actions = list()
            for idx in range(len(avail_action_list)):
                avail_action = np.array(avail_action_list[idx])
                avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                actions.append(
                    np.random.choice(avail_actions_index, p=softmax(target_distance_list[idx], temperature=0)))

        if self.rule == 'rule2':
            actions = list()
            if air_alert == True:
                #print(self.temperature1, target_distance_list[0], softmax(target_distance_list[0], temperature = self.temperature1))
                #print(self.temperature2, target_distance_list[0], softmax(target_distance_list[0], temperature = self.temperature2))
                for idx in range(len(avail_action_list)):
                    avail_action = np.array(avail_action_list[idx])
                    avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                    actions.append(np.random.choice(avail_actions_index, p = softmax(target_distance_list[idx], temperature = self.temperature1, debug = True)))
            else:
                for idx in range(len(avail_action_list)):
                    avail_action = np.array(avail_action_list[idx])
                    avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                    actions.append(np.random.choice(avail_actions_index, p = softmax(target_distance_list[idx], temperature = self.temperature2, debug = True)))
        return actions



