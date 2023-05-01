from Components.Modeler_Component import *
from Components.Adapter_Component import *
from Components.Policy import *
from collections import deque
from cfg import get_cfg
from GDN import Agent
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

def train(agent, env, e, t, train_start, epsilon, min_epsilon, anneal_epsilon, initializer, output_dir, vdn, n_step):
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
    dummy_avail_action = [[False]*agent.action_size for _ in range(agent.num_agent)]
    n_step_rewards = deque(maxlen = n_step)
    n_step_dones = deque(maxlen = n_step)
    n_step_missile_node_features =deque(maxlen = n_step)
    n_step_ship_feature = deque(maxlen = n_step)
    n_step_edge_index = deque(maxlen = n_step)
    n_step_action_blue =deque(maxlen = n_step)
    n_step_avail_action_blue = deque(maxlen = n_step)
    step_checker = 0
    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')

            ship_feature = env.get_ship_feature()
            edge_index   = env.get_edge_index()
            missile_node_feature = env.get_missile_node_feature()
            n_node_feature_machine =env.friendlies_fixed_list[0].surface_tracking_limit+ env.friendlies_fixed_list[0].air_tracking_limit+1

            agent.eval_check(eval=True)
            node_representation = agent.get_node_representation(missile_node_feature, ship_feature,edge_index,n_node_feature_machine,mini_batch=False)  # 차원 : n_agents X n_representation_comm
            action_blue = agent.sample_action(node_representation, avail_action_blue, epsilon)

            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
            reward, win_tag, done = env.step(action_blue, action_yellow)
            #step+=1
            episode_reward += reward

            n_step_missile_node_features.append(missile_node_feature)
            n_step_ship_feature.append(ship_feature)
            n_step_edge_index.append(edge_index)
            n_step_action_blue.append(action_blue)
            n_step_rewards.append(reward)
            n_step_dones.append(done)
            n_step_avail_action_blue.append(avail_action_blue)

            status = None
            step_checker += 1
            if step < (n_step-1):
                step += 1
            else:
                idx = (n_step-1)-step
                agent.buffer.memory(n_step_missile_node_features[idx],
                                    n_step_ship_feature[idx],
                                    n_step_edge_index[idx],
                                    n_step_action_blue[idx],
                                    n_step_rewards,
                                    n_step_dones,
                                    n_step_avail_action_blue[idx],
                                    status)



            if e >= train_start:
                t += 1
                if agent.beta <= 1:
                    agent.beta += 0.0005
                agent.eval_check(eval=False)
                agent.learn(regularizer=0, vdn=vdn)
        else:
            pass_transition = True
            env.step(action_blue=friendly_action_for_transition,
                                                    action_yellow=enemy_action_for_transition, pass_transition = pass_transition)

        if done == True:
            if step_checker < n_step:



                while len(n_step_rewards) < n_step:
                    n_step_dones.append(True)
                    n_step_rewards.append(0)
                    n_step_action_blue.append([0] * agent.num_agent)
                    n_step_avail_action_blue.append(dummy_avail_action)
                    n_step_missile_node_features.append(np.zeros_like(missile_node_feature).tolist())
                    n_step_ship_feature.append(np.zeros_like(ship_feature).tolist())
                    n_step_edge_index.append([[], []])

                agent.buffer.memory(n_step_missile_node_features[0],
                                    n_step_ship_feature[0],
                                    n_step_edge_index[0],
                                    n_step_action_blue[0],
                                    n_step_rewards,
                                    n_step_dones,
                                    n_step_avail_action_blue[0],
                                    status)



            for i in range(step-1):

                n_step_dones.append(True)
                n_step_rewards.append(0)
                #print(n_step_avail_action_blue[0])
                agent.buffer.memory(n_step_missile_node_features[0],
                                    n_step_ship_feature[0],
                                    n_step_edge_index[0],
                                    n_step_action_blue[0],
                                    n_step_rewards,
                                    n_step_dones,
                                    n_step_avail_action_blue[0],
                                    status)

                n_step_action_blue.append([0]*agent.num_agent)
                n_step_avail_action_blue.append(dummy_avail_action)
                n_step_missile_node_features.append(np.zeros_like(missile_node_feature).tolist())
                n_step_ship_feature.append(np.zeros_like(ship_feature).tolist())
                n_step_edge_index.append([[],[]])

            break
    return episode_reward, epsilon, t, eval



if __name__ == "__main__":
    cfg = get_cfg()
    vessl_on = cfg.vessl
    if vessl_on == True:
        import vessl

        vessl.init()
        output_dir = "/output/"
    else:
        from torch.utils.tensorboard import SummaryWriter
        output_dir = "../output_susceptibility/"
        writer = SummaryWriter('./logs2')



    import time
    """
    환경 시스템 관련 변수들
    """
    visualize = False           # 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
    size = [600, 600]              # 화면 size / 600, 600 pixel
    tick = 500                     # 가시화 기능 사용 시 빠르기
    n_step = cfg.n_step
    simtime_per_frame = cfg.simtime_per_frame
    decision_timestep = cfg.decision_timestep
    detection_by_height = False      # 고도에 의한
    num_iteration = cfg.num_episode          # 시뮬레이션 반복횟수
    mode = 'txt'                 # 전처리 모듈 / 'excel' : input_data.xlsx 파일 적용, 'txt' "Data\ship.txt", "Data\patrol_aircraft.txt", "Data\SAM.txt", "Data\SSM.txt"를 적용
    rule = 'rule2'               # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)
    temperature = [10, 20]       # rule = 'rule2'인 경우만 적용 / 의사결정의 flexibility / 첫번째 index : 공중 위험이 낮은 상태, 두번째 index : 공중 위험이 높은 상태
    ciws_threshold = 1
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
    import torch, random
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

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
                  feature_size_job=env.get_env_info()["job_feature_shape"],
                  feature_size_machine=env.get_env_info()["machine_feature_shape"],
                  iqn_layers=cfg.iqn_layers,
                  node_embedding_layers_job=cfg.job_layers,
                  node_embedding_layers_machine=cfg.machine_layers,
                  hidden_size_comm = cfg.hidden_size_comm,
                  n_multi_head=cfg.n_multi_head,
                  n_representation_job=cfg.n_representation_job,
                  n_representation_machine=cfg.n_representation_machine,
                  dropout=0.6,
                  action_size=env.get_env_info()["n_actions"],
                  buffer_size=cfg.buffer_size,
                  batch_size=cfg.batch_size,
                  learning_rate=cfg.lr,#0.0001,
                  gamma=cfg.gamma,
                  GNN='GAT',
                  teleport_probability=cfg.teleport_probability,
                  gtn_beta=0.1,
                  n_node_feature = env.friendlies_fixed_list[0].surface_tracking_limit+env.friendlies_fixed_list[0].air_tracking_limit+1,
                  n_step= n_step,
                  beta = cfg.per_beta)
    anneal_steps = 50000
    epsilon = 1
    min_epsilon = 0.01
    #anneal_epsilon = (epsilon - min_epsilon) / anneal_steps
    reward_list = list()

    for e in range(num_iteration):
        start = time.time()


        #print("소요시간", time.time()-start)
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold)


        episode_reward, epsilon, t, eval = train(agent, env, e, t, train_start=cfg.train_start, epsilon=epsilon, min_epsilon=min_epsilon, anneal_epsilon=0 , initializer=False, output_dir=None, vdn=True, n_step = n_step)
        writer.add_scalar("episode", episode_reward, e)
        reward_list.append(episode_reward)
        if e % 10 == 0:
            import os
            import pandas as pd
            output_dir = "../output_dir_susceptibility/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df = pd.DataFrame(reward_list)
            df.to_csv(output_dir + 'episode_reward.csv')

        if e % 500 == 0:
            agent.save_model(e, t, epsilon, output_dir + "{}.pt".format(e))

        #print(len(agent.buffer.buffer[2]))
        print(
            "Total reward in episode {} = {}, epsilon : {}, time_step : {}, episode_duration : {}".format(
                e,
                np.round(episode_reward, 3),
                np.round(epsilon, 3),
                t, np.round(time.time() - start, 3)))

        # del data
        # del env