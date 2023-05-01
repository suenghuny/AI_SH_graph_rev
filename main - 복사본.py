import numpy as np
import pandas as pd
from Components.Modeler_Component import *
from Components.Adapter_Component import *
from Components.Reporter_Component import *
from Components.Policy import *
from copy import deepcopy

import pandas as pd
from GDN import Agent
from functools import partial
import numpy as np


def preprocessing(scenarios):
    scenario = scenarios[0]
    if mode == 'txt':
        input_path = ["Data\{}\ship.txt".format(scenario),
                      "Data\{}\patrol_aircraft.txt".format(scenario),
                      "Data\{}\SAM.txt".format(scenario),
                      "Data\{}\SSM.txt".format(scenario),
                      "Data\{}\inception.txt".format(scenario)]
    else:
        input_path = "Data\input_data.xlsx"

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data

def train(agent, env, e, t, train_start, epsilon, min_epsilon, anneal_epsilon, initializer, output_dir, vdn):
    agent_yellow = Policy(env, rule='rule2', temperatures=temperature)
    done = False
    episode_reward = 0
    step = 0
    losses = []
    epi_r = list()
    eval = False

    sum_learn = 0
    enemy_action_for_transition =    [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            ship_feature = env.get_ship_feature()
            edge_index   = env.get_edge_index()
            missile_node_feature = env.get_missile_node_feature()
            n_node_feature_machine = env.friendlies_fixed_list[0].air_tracking_limit+1

            agent.eval_check(eval=True)
            node_representation = agent.get_node_representation(missile_node_feature, ship_feature,edge_index,n_node_feature_machine,mini_batch=False)  # 차원 : n_agents X n_representation_comm
            action_blue = agent.sample_action(node_representation, avail_action_blue, epsilon)

            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
            reward, win_tag, done = env.step(action_blue, action_yellow)

            episode_reward += reward

            status = None
            agent.buffer.memory(missile_node_feature, ship_feature, edge_index, action_blue, reward, done, avail_action_blue, status)

            if e >= train_start:
                t += 1
                if epsilon >= min_epsilon:
                    epsilon = epsilon - anneal_epsilon
                else:
                    epsilon = min_epsilon
                agent.eval_check(eval=False)
                agent.learn(regularizer=0, vdn=vdn)
        else:
            pass_transition = True
            env.step(action_blue=friendly_action_for_transition,
                                                    action_yellow=enemy_action_for_transition, pass_transition = pass_transition)

        if done == True:
            agent.eval_check(eval=False)
            loss = agent.learn(regularizer=0, vdn=vdn)
            losses.append(loss.detach().item())
            break
    return episode_reward, epsilon, t, eval



if __name__ == "__main__":




    import time

    #start = time.time()
    """
    시뮬레이션 기본 입력값 준비
    """
    visualize = True            # 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
    size = [600, 600]              # 화면 size / 600, 600 pixel
    tick = 2                      # 가시화 기능 사용 시 빠르기
    simtime_per_frame = 2
    decision_timestep = 5
    detection_by_height = False      # 고도에 의한
    num_iteration = 100000           # 시뮬레이션 반복횟수
    mode = 'txt'                 # 전처리 모듈 / 'excel' : input_data.xlsx 파일 적용, 'txt' "Data\ship.txt", "Data\patrol_aircraft.txt", "Data\SAM.txt", "Data\SSM.txt"를 적용
    rule = 'rule2'               # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)
    temperature = [10, 20]       # rule = 'rule2'인 경우만 적용 / 의사결정의 flexibility / 첫번째 index : 공중 위험이 낮은 상태, 두번째 index : 공중 위험이 높은 상태
    ciws_threshold = 2.5
    polar_chart_visualize = False
    scenarios = ['scenario1', 'scenario2', 'scenario3']

    lose_ratio = list()
    remains_ratio = list()

    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 50]  # RCS의 polarchart 적용

    polar_chart = [polar_chart_scenario1]
    df_dict = {}

    #scenario = np.random.choice(scenarios)

    episode_polar_chart = polar_chart[0]
    records = list()
    np.random.seed(1230)

    data = preprocessing(scenarios)
    t = 0
    env = modeler(data,
                  visualize=visualize,
                  size=size,
                  detection_by_height=detection_by_height,
                  tick=tick,
                  simtime_per_framerate=simtime_per_frame,
                  ciws_threshold=ciws_threshold)

    agent = Agent(num_agent=1,
                  feature_size_job=10,
                  feature_size_machine=6,
                  hidden_size_meta_path=5,
                  hidden_size_obs=64,
                  hidden_size_comm=64,
                  hidden_size_Q=128,
                  n_multi_head=1,
                  n_representation_job=24,
                  n_representation_machine=12,
                  dropout=0.6,
                  action_size=env.get_env_info()["n_actions"],
                  buffer_size=50000,
                  batch_size=32,
                  learning_rate=0.0001,
                  gamma=0.99,
                  GNN='GAT',
                  teleport_probability=0.1,
                  gtn_beta=0.1,
                  n_node_feature = env.friendlies_fixed_list[0].air_tracking_limit+1)
    anneal_steps = 50000
    epsilon = 0.5
    min_epsilon = 0.01
    anneal_epsilon = (epsilon - min_epsilon) / anneal_steps

    for e in range(num_iteration):
        #
        # if e == 20:
        #     visualize = True
        # else:
        #     visualize = False
        #
        start = time.time()


        #print("소요시간", time.time()-start)
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold)


        episode_reward, epsilon, t, eval = train(agent, env, e, t, 2, epsilon=epsilon, min_epsilon=min_epsilon, anneal_epsilon=anneal_epsilon , initializer=False, output_dir=None, vdn=True)
        #print(len(agent.buffer.buffer[2]))
        print(
            "Total reward in episode {} = {}, epsilon : {}, time_step : {}, episode_duration : {}".format(
                e,
                np.round(episode_reward, 3),
                np.round(epsilon, 3),
                t, np.round(time.time() - start, 3)))

        # del data
        # del env