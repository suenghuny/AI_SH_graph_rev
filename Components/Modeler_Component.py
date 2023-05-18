from Components.Adapter_Component import *
from Components.Simulation_Component import *
from collections import deque


def modeler(data, visualize, size, detection_by_height, tick, simtime_per_framerate, ciws_threshold, action_history_step, epsilon = 20):
    env = Environment(data,
                      visualize,
                      size = size,
                      detection_by_height = detection_by_height,
                      tick = tick,
                      simtime_per_framerate = simtime_per_framerate,
                      epsilon = epsilon,
                      ciws_threshold = ciws_threshold,
                      action_history_step = action_history_step)
    return env

class Environment:
    def __init__(self,
                 data,
                 visualize,
                 action_history_step,
                 epsilon=15,
                 simtime_per_framerate = 2.5,
                 size = [2200, 2200],
                 detection_by_height = True,
                 tick = 24,
                 ciws_threshold = 2.5,
                 mode = False):
        self.simtime_per_framerate = simtime_per_framerate # 시뮬레이션 시간 / 프레임 주기
        self.nautical_mile_scaler = self.simtime_per_framerate / 3600 * 10
        self.detection_by_height = detection_by_height
        self.mach_scaler = self.simtime_per_framerate          / 3600 * 10 * 660.907127
        self.F = 10**0.3           # noise factor
        self.F_surface = 10**0.3   # noise factor
        self.L = 0.5               # constant of loss for reflection
        self.k = 1.38 * 10**-23    # boltzmann constant
        self.temperature = 290     # temperature
        self.P_fa = 10**-7         # error of detection
        self.P_fa_surface = 10**-7 # error of detection
        self.size = size           # SCALE : 60NM, 60NM 공간
        self.now = 0
        self.visualize = visualize
        self.temp_termination = False
        self.tick = tick
        self.action_history_step = action_history_step
        if visualize == True:
            self.pygame = pygame
            self.game_initializer = self.pygame.init()
            self.clock = pygame.time.Clock()
            self.screen = self.pygame.display.set_mode(self.size)
            self.title = "Suceptibility Evaluation"
            self.pygame.display.set_caption(self.title)

        self.ref_coordinate = [self.size[0] / 2, self.size[1] / 2]
        self.black = (0, 0, 0)
        self.data = data

        self.last_destroyed_missile = 0
        self.last_destroyed_enemy = 0
        self.last_destroyed_ship = 0



        self.ships = list()
        self.patrol_aircrafts = list()
        self.epsilon = epsilon
        self.ciws_threshold = ciws_threshold

        # for key, value in data.patrol_aircraft_data.items():
        #     self.patrol_aircrafts.append(Patrol_aircraft(env=self,
        #                                                  id=key,
        #                                                  speed=value['speed'],
        #                                                  course=value['course'],
        #                                                  radius=value['radius'],
        #                                                  initial_position_x=value['position_x']*10,
        #                                                  initial_position_y=value['position_y']*10,
        #                                                  side=value['side']
        #                                                  ))

        self.num_friendly_ssm = sum([value['num_ssm'] for key, value in data.ship_data.items() if value['side'] == 'blue'])
        self.num_enemy_ssm = sum([value['num_ssm'] for key, value in data.ship_data.items() if value['side'] == 'yellow'])
        #ship_data, enemy_data = self.get_position_inception(self, self.data.inception_data, num_enemy)

        inception_data = self.data.inception_data
        noise = np.random.uniform(-10, 10)

        self.missile_speed_list = list()

        for key, value in data.ship_data.items():
            if key == 1:
                speed = 25
                course = 90
                initial_position_x = 50
                initial_position_y = 50
            else:
                if mode == True:
                    speed = 25
                    course = 90
                    initial_position_x = 50 + 10*inception_data['inception_distance'] * np.cos(inception_data['inception_angle'] * np.pi / 180) + 10*np.random.normal(inception_data['enemy_spacing_mean'], inception_data['enemy_spacing_std'])
                    initial_position_y = 50 + 10*inception_data['inception_distance'] * np.sin(inception_data['inception_angle'] * np.pi / 180) + 10*np.random.normal(inception_data['enemy_spacing_mean'], inception_data['enemy_spacing_std'])
                else:
                    speed = 25
                    course = 90
                    initial_position_x = 50 + 10 * inception_data['inception_distance'] * np.cos(
                        inception_data['inception_angle'] * np.pi / 180) + 10 * np.random.normal(
                        inception_data['enemy_spacing_mean'], inception_data['enemy_spacing_std'])
                    initial_position_y = 50 + 10 * inception_data['inception_distance'] * np.sin(
                        inception_data['inception_angle'] * np.pi / 180) + 10 * np.random.normal(
                        inception_data['enemy_spacing_mean'], inception_data['enemy_spacing_std'])

                    # speed = 25
                    # course = 90
                    # initial_position_x = 50 + 10*inception_data['inception_distance'] * np.cos((inception_data['inception_angle']+noise) * np.pi / 180)+10*np.random.normal(inception_data['enemy_spacing_mean'], inception_data['enemy_spacing_std'])
                    # initial_position_y = 50 + 10*inception_data['inception_distance'] * np.sin((inception_data['inception_angle']+noise) * np.pi / 180)+10*np.random.normal(inception_data['enemy_spacing_mean'], inception_data['enemy_spacing_std'])



            type_m_sam = data.SAM_data[int(value['type_m_sam'])]
            type_l_sam = data.SAM_data[int(value['type_l_sam'])]
            type_ssm = data.SSM_data[int(value['type_ssm'])]
            self.missile_speed_list.append((type_ssm['speed']+type_l_sam['speed'])*self.mach_scaler)
            self.missile_speed_list.append((type_ssm['speed']+type_m_sam['speed'])*self.mach_scaler)
            #self.missile_speed_list.append(type_m_sam['speed'])
            self.ships.append(Ship(env=self,
                                   id=key,
                                   speed=speed,
                                   course=course,
                                   initial_position_x=initial_position_x,
                                   initial_position_y=initial_position_y,
                                   type_m_sam=type_m_sam,
                                   num_m_sam=value['num_m_sam'],

                                   type_l_sam=type_l_sam,
                                   num_l_sam=value['num_l_sam'],

                                   type_ssm=type_ssm,
                                   num_ssm=value['num_ssm'],

                                   length=value['length'],
                                   breadth=value['breadth'],
                                   height=value['height'],
                                   surface_engagement_limit=value['surface_engagement_limit'],
                                   surface_tracking_limit=value['surface_tracking_limit'],
                                   air_tracking_limit=value['air_tracking_limit'],
                                   air_engagement_limit=value['air_engagement_limit'],
                                   detection_range=value['detection_range'],
                                   ciws_max_range = value['ciws_max_range'],

                                   decoy_launching_distance=value['decoy_launching_distance'],
                                   decoy_launching_bearing=value['decoy_launching_bearing'],
                                   decoy_launching_interval = value['decoy_launching_interval'],

                                   evading_course=value['evading_course'],
                                   side=value['side'],

                                   radar_peak_power = value['radar_peak_power'],
                                   antenna_gain_factor = value['antenna_gain_factor'],
                                   wavelength_of_signal = value['wavelength_of_signal'],
                                   radar_receiver_bandwidth=value['radar_receiver_bandwidth'],
                                   decoy_rcs=value['decoy_rcs'],

                                   decoy_duration = value['decoy_duration'],
                                   decoy_decaying_rate = value['decoy_decaying_rate'],
                                   ciws_max_num_per_min = value['ciws_max_num_per_min'],
                                   ciws_bullet_capacity = value['ciws_bullet_capacity'],

                                   ssm_launching_duration_min = value['ssm_launching_duration_min'],
                                   ssm_launching_duration_max = value['ssm_launching_duration_max'],
                                   lsam_launching_duration_min=value['lsam_launching_duration_min'],
                                   lsam_launching_duration_max=value['lsam_launching_duration_max'],
                                   msam_launching_duration_min=value['msam_launching_duration_min'],
                                   msam_launching_duration_max=value['msam_launching_duration_max'],
                                   interpolating_rcs=data.get_rcs
                                   ))

        self.friendlies = [ship for ship in self.ships if ship.side == 'blue']  # 모든 함정은 접촉을 유지하고 있다고 가정함
        self.enemies = [ship for ship in self.ships if ship.side == 'yellow']  # 모든 함정은 접촉을 유지하고 있다고 가정함
        self.missile_speed_scaler = np.max(self.missile_speed_list)
        self.friendlies_fixed_list = [ship for ship in self.ships if ship.side == 'blue']
        self.enemies_fixed_list = [ship for ship in self.ships if ship.side == 'yellow']

        self.friendlies_fixed_patrol_aircraft_list = [patrol_aircraft for patrol_aircraft in self.patrol_aircrafts if patrol_aircraft.side == 'blue']
        self.enemies_fixed_patrol_aircraft_list = [patrol_aircraft for patrol_aircraft in self.patrol_aircrafts if patrol_aircraft.side == 'yellow']



        self.ship_friendly_action_space = np.max([ship.surface_tracking_limit for ship in self.friendlies])
        self.ship_enemy_action_space = np.max([ship.surface_tracking_limit for ship in self.enemies])

        self.air_friendly_action_space = np.max([ship.air_tracking_limit for ship in self.friendlies])
        self.air_enemy_action_space = np.max([ship.air_tracking_limit for ship in self.enemies])

        self.flying_ssms_enemy = list()
        self.flying_sams_enemy = list()
        self.flying_ssms_friendly = list()
        self.flying_sams_friendly = list()

        self.decoys_friendly = list()
        self.decoys_enemy = list()

        self.action_size_enemy = (1 + self.ship_enemy_action_space + self.air_enemy_action_space)
        self.action_size_friendly = (1 + self.ship_friendly_action_space + self.air_friendly_action_space)

        self.avail_action_enemy = [False] * (1 + self.ship_enemy_action_space + self.air_enemy_action_space)
        self.avail_action_friendly = [False] * (1 + self.ship_friendly_action_space + self.air_friendly_action_space)

        self.event_log = list()
        self.debug_monitors = list()

        self.temp_max_air_engagement = list()
        self.temp_rcs = list()
        self.f7 = 0
        self.f8 = 0
        self.f9 = 0
        self.f10 = 0
        self.last_action_encodes = np.eye(self.action_size_friendly)
        self.f11 = [0,0,0,0,0,0,0,0]
        self.f11_deque = deque(maxlen = self.action_history_step)
        for _ in range(self.action_history_step):
            self.f11_deque.append(self.f11)

    def get_env_info(self):
        env_info = {"n_agents" : 1,
                    "ship_feature_shape": 10+1+8,  # + self.n_agents,
                    "missile_feature_shape" : 6,  #9 + num_jobs + max_ops_length+ len(workcenter)+3+len(ops_name_list) + 1+3-12, # + self.n_agents,
                    "enemy_feature_shape": 12,
                    "action_feature_shape": 8,
                    # 9 + num_jobs + max_ops_length+ len(workcenter)+3+len(ops_name_list) + 1+3-12, # + self.n_agents,
                    "n_actions": (1 + self.ship_friendly_action_space + self.air_friendly_action_space)
                    }
        return env_info



    def get_target_availability(self, friendlies_fixed_list, avail_action_friendly_model, enemies_fixed_list, flying_ssms_enemies, interval_min, interval_constant):
        avail_actions = list()
        target_distance_list = list()
        air_alert = False

        for ship in friendlies_fixed_list:
            avail_action_friendly = deepcopy(avail_action_friendly_model)
            len_avail_action_friendly = len(avail_action_friendly)
            avail_action_friendly[0] = True  # null-action은 항상 True
            ship.get_detections()
            distance_list = list()
            if ship.status != 'destroyed':  # destroy 되지 않았을 때만 다른 행동을 수행할 수 있음
                " 수상함 공격에 대한 부분 "
                " 파괴되지 않았으며, ssm의 사거리 이내에 있는 표적에 대해 공격가능"
                " 수상표적은 빠르지 않으며, (유도탄에 비해) 수가 빠르게 변화하지는 않기 때문에 action-index에 one-to-one 매핑되도록함"
                if len(ship.surface_prelaunching_managing_list) < ship.surface_engagement_limit:
                    num_of_idle_ssms = len([1 for missile in ship.ssm_launcher if missile.status == 'idle'])
                    if num_of_idle_ssms > 0:  # 보유 ssm 대수가 1발 이상이어야 수상표적 공격 가능
                        for i in range(len(enemies_fixed_list)):              # 함정 표적은 feature index가 정해져 있음
                            enemy_ship = enemies_fixed_list[i]
                            d = cal_distance(ship, enemy_ship)
                            if enemy_ship.status != 'destroyed':              # 해당 적함의 상태가 destroy가 아니어야 공격가능
                                if d <= ship.ssm_max_range:                   # 해당 적함의 거리가 ssm의 사거리 내에 있어야함
                                    avail_action_friendly[i + 1] = True       # null-action의 index가 0이므로 그 이후부터 채워 나감감
                                    distance_list.append(d)

                " 대공표적 공격에 대한 부분 "
                " sam의 경우는 거리에 따라서 index를 지정함(숫자가 많고 변동성이 다소 심하므로 고정된 index를 가지기 어려움)"
                " 거리가 가까운 순서대로 앞에 index를 가짐"
                " action_index : [no-ops, ship1, ship2, ship3, ... surface_tracking_limit, air1, air2, ..... air_tracking_limit]"
                " 거리 : air1 <= air2 <= air3...."
                if len(ship.air_engagement_managing_list) < ship.air_engagement_limit:
                    idle_l_sam = [l_sam for l_sam in ship.l_sam_launcher if
                                  l_sam.status == 'idle']                 # sam의 경우 동시 공격이 가능하므로, idle 상태인 유도탄 전체에 대한 정보가 필요함(해당 정보를 생성하는 부분)
                    idle_m_sam = [m_sam for m_sam in ship.m_sam_launcher if
                                  m_sam.status == 'idle']                 # sam의 경우 동시 공격이 가능하므로, idle 상태인 유도탄 전체에 대한 정보가 필요함(해당 정보를 생성하는 부분)
                    distance_range_1 = [cal_distance(ship, ssm) for ssm in ship.ssm_detections if
                               cal_distance(ship, ssm) > ship.l_sam_max_range]                                                # 1구간 : > l_sam range
                    distance_range_2 = [cal_distance(ship, ssm) for ssm in ship.ssm_detections
                               if (cal_distance(ship, ssm) <= ship.l_sam_max_range) and (cal_distance(ship, ssm) > ship.m_sam_max_range)]  # 2구간 : <= l_sam_range, > m_sam_range

                    distance_range_3 = [cal_distance(ship, ssm) for ssm in ship.ssm_detections
                               if (cal_distance(ship, ssm) <= ship.m_sam_max_range) and (cal_distance(ship, ssm) > ship.ciws_max_range)]  # 3구간 : <= m_sam_range, > ciws_range

                    distance_range_4 = [cal_distance(ship, ssm) for ssm in ship.ssm_detections
                               if cal_distance(ship, ssm) <= ship.ciws_max_range]

                    for ssm in ship.ssm_detections:
                        if (cal_distance(ship, ssm) <= 200):
                            air_alert = True
                            break
                        else:
                            air_alert = False

                    # 4구간 : <= ciws_range
                    len1 = len(distance_range_1)
                    len2 = len(distance_range_2)
                    len3 = len(distance_range_3)
                    len4 = len(distance_range_4)



                    len_idle_l_sam = len(idle_l_sam)
                    len_idle_m_sam = len(idle_m_sam)



                    if (len2 == 0) and (len3 == 0) and (len4 == 0):  # 모든 구간에서 0개 표적
                        pass
                    else:
                        if (len2 != 0) and (len3 != 0) and (len4 != 0):  # 모든 구간에서 한개 이상의 target이 존재
                            if len_idle_l_sam > 0:  # l_sam 잔여량 1개 이상
                                concat_distance = np.concatenate([distance_range_4, distance_range_3, distance_range_2])
                                for s in range(len4 + len3 + len2):  # 모든 범위(range_2, range_3, range_4) 내에 표적 교전 가능
                                    if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                        avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                        distance_list.append(concat_distance[s])

                            if (len_idle_l_sam == 0) and (len_idle_m_sam > 0):  # l_sam 잔여량은 0개, m_sam의 잔여량은 1개 이상
                                concat_distance = np.concatenate([distance_range_4, distance_range_3])
                                for s in range(len4 + len3):  # m_sam 및 ciws 범위(range_3, range_4) 내에 표적 교전 가능
                                    if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                        avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                        distance_list.append(concat_distance[s])

                            if (len_idle_l_sam == 0) and (len_idle_m_sam == 0):  # l_sam, m_sam 모두 잔여량 없음(ciws로만 교전 가능)
                                concat_distance = np.concatenate([distance_range_4])
                                for s in range(len4):  # sam 교전 불가, ciws로만 교전 가능
                                    if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                        avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                        distance_list.append(concat_distance[s])



                        else:  # 적어도 한개 구간에서 0개 표적 이상
                            if (len2 == 0) and (len3 != 0) and (len4 != 0):  # m_sam 및 ciws 교전구역에 표적 존재
                                concat_distance = np.concatenate([distance_range_4, distance_range_3])
                                if (len_idle_l_sam > 0):  # l_sam 잔여량 한개 이상
                                    for s in range(len3 + len4):  # m_sam 및 ciws 범위(range_3, range_4) 내에 표적 교전 가능
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam > 0):  # l_sam 잔여량 0개, m_sam 잔여량 한개 이상
                                    concat_distance = np.concatenate([distance_range_4, distance_range_3])
                                    for s in range(len3 + len4):  # m_sam 및 ciws 범위(range_3, range_4) 내에 표적 교전 가능
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam == 0):  # l_sam 및 m_sam 잔여량 0개
                                    concat_distance = np.concatenate([distance_range_4])
                                    for s in range(len4):  # ciws 범위(range-4) 교전 가능 / sam 교전 불가
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                            if (len2 != 0) and (len3 == 0) and (len4 != 0):  # l_sam 및 ciws 교전구역에 표적 존재
                                concat_distance = np.concatenate([distance_range_4, distance_range_2])
                                if (len_idle_l_sam > 0):  # l_sam 잔여량 한개 이상
                                    for s in range(len2 + len4):  # l_sam 및 ciws 범위(range_2, range-4) 내에 표적 교전 가능
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam > 0):  # l_sam 잔여량 0개, m_sam 잔여량 한개 이상
                                    concat_distance = np.concatenate([distance_range_4])
                                    for s in range(len4):  # ciws 범위(range_4) 내에 표적 교전 가능
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam == 0):  # l_sam 및 m_sam 잔여량 0개
                                    concat_distance = np.concatenate([distance_range_4])
                                    for s in range(len4):  # ciws 범위(range_4) 교전 가능 / sam 교전 불가
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                            if (len2 != 0) and (len3 != 0) and (len4 == 0):  # l_sam 및 m_sam 교전구역에 표적 존재

                                if (len_idle_l_sam > 0):  # l_sam 잔여량 한개 이상
                                    concat_distance = np.concatenate([distance_range_3, distance_range_2])
                                    for s in range(len2 + len3):  # l_sam 및 m_sam 교전구역 공격 가능
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam > 0):  # l_sam 잔여량 0개, m_sam 잔여량 한개 이상
                                    concat_distance = np.concatenate([distance_range_3])
                                    for s in range(len3):  # m_sam 교전구역 공격 가능
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam == 0):  # l_sam 및 m_sam 잔여량 0개
                                    pass  # 교전 가능 표적 없음

                            if (len2 != 0) and (len3 == 0) and (len4 == 0):  # l_sam 교전구역에 표적 존재
                                if (len_idle_l_sam > 0):  # l_sam 잔여량 한개 이상
                                    concat_distance = np.concatenate([distance_range_2])
                                    for s in range(len2):  # l_sam 교전구역 공격 가능
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam > 0):  # l_sam 잔여량 0개, m_sam 잔여량 한개 이상
                                    pass  # 교전 가능 표적 없음

                                if (len_idle_l_sam == 0) and (len_idle_m_sam == 0):  # l_sam 및 m_sam 잔여량 0개
                                    pass  # 교전 가능 표적 없음

                            if (len2 == 0) and (len3 != 0) and (len4 == 0):  # m_sam 교전구역에 표적 존재
                                if (len_idle_l_sam > 0):  # l_sam 잔여량 한개 이상
                                    concat_distance = np.concatenate([distance_range_3])
                                    for s in range(len3):  # m_sam 교전구역 공격 가능
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam > 0):  # l_sam 잔여량 0개, m_sam 잔여량 한개 이상
                                    concat_distance = np.concatenate([distance_range_3])
                                    for s in range(len3):  # m_sam 교전구역 공격 가능
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam == 0):  # l_sam 및 m_sam 잔여량 0개
                                    pass  # 교전 가능 표적 없음

                            if (len2 == 0) and (len3 == 0) and (len4 != 0):  # ciws 교전구역 표적 존재
                                if (len_idle_l_sam > 0):  # l_sam 잔여량 한개 이상
                                    concat_distance = np.concatenate([distance_range_4])
                                    for s in range(len4):  # ciws 범위(range_4) 교전 가능 / sam 교전 불가
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam > 0):  # l_sam 잔여량 0개, m_sam 잔여량 한개 이상
                                    concat_distance = np.concatenate([distance_range_4])
                                    for s in range(len4):  # ciws 범위(range_4) 교전 가능 / sam 교전 불가
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])

                                if (len_idle_l_sam == 0) and (len_idle_m_sam == 0):  # l_sam 및 m_sam 잔여량 0개
                                    concat_distance = np.concatenate([distance_range_4])
                                    for s in range(len4):  # ciws 범위(range_4) 교전 가능 / sam 교전 불가
                                        if s + ship.surface_tracking_limit + 1 < len_avail_action_friendly:
                                            avail_action_friendly[s + ship.surface_tracking_limit + 1] = True
                                            distance_list.append(concat_distance[s])
            avail_actions.append(avail_action_friendly)
            if distance_list==list():
                distance_list.append(1)
            else:
                if interval_min == True:
                    distance_list.insert(0, np.min(distance_list)/interval_constant)
                else:
                    distance_list.insert(0, np.min(distance_list)/interval_constant)
            target_distance_list.append(distance_list)
        return avail_actions, target_distance_list, air_alert

    def get_avail_actions_temp(self, interval_min, interval_constant, side='blue'):
        if side != 'blue':
            avail_actions, target_distance_list, air_alert = self.get_target_availability(self.enemies_fixed_list, self.avail_action_enemy, self.friendlies_fixed_list, self.flying_ssms_friendly, interval_min, interval_constant)
        else:
            avail_actions, target_distance_list, air_alert = self.get_target_availability(self.friendlies_fixed_list, self.avail_action_friendly, self.enemies_fixed_list, self.flying_ssms_enemy, interval_min, interval_constant)
        return avail_actions, target_distance_list, air_alert

    def get_ship_feature(self):
        ship_feature = list()
        enemy_ssm = 0
        enemy_ship = 0
        for enemy in self.enemies_fixed_list:
            enemy_ship +=1
            enemy_ssm += enemy.num_ssm
        for ship in self.friendlies_fixed_list:
            f1 = len(ship.air_engagement_managing_list)/ship.air_engagement_limit
            f2 = len(ship.air_prelaunching_managing_list)/ship.air_engagement_limit
            f3 = len(ship.surface_prelaunching_managing_list)/ship.surface_engagement_limit
            f4 = len(ship.m_sam_launcher)/ship.num_m_sam
            f5 = len(ship.l_sam_launcher) /ship.num_l_sam
            f6 = len(ship.ssm_launcher)/ship.num_ssm
            f7 = self.f7 / enemy_ssm
            f8 = self.f8 / enemy_ship
            f9 = self.f9 / self.simtime_per_framerate
            f10 = self.f10 / self.simtime_per_framerate
            f11 = self.f11
            f12 = ship.health /ship.init_health
            ship_feature.append(np.concatenate([[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10, f12], f11]).tolist())
        #print(len(f11), len(ship_feature[0]),"dddd")
        return ship_feature

    def get_edge_index(self):
        edge_index = [[],[]]
        ship = self.friendlies_fixed_list[0]
        len_ssm_detections = len(ship.ssm_detections)
        len_flying_sams_friendly = len(self.flying_sams_friendly)
        # print("==================")
        # print("ssm 길이", len_ssm_detections)
        # print("sam 길이", len_flying_sams_friendly)
        for i in range(1, len_ssm_detections+1):
            edge_index[0].append(0)
            edge_index[1].append(i)
            #print("전", 0, i, "전체 길이", 1+len_ssm_detections+len_flying_sams_friendly)


        for i in range(1, len_ssm_detections+1):
            for j in range(len_flying_sams_friendly):
                #print("후", i, len_ssm_detections+j+1, "전체 길이", 1+len_ssm_detections+len_flying_sams_friendly)

                ssm_i = ship.ssm_detections[i-1]
                sam_j = self.flying_sams_friendly[j]

                if sam_j.original_target==ssm_i:
                    edge_index[0].append(i)
                    edge_index[1].append(len_ssm_detections+j+1)

                    print(i, len_ssm_detections+j+1)



        #     for k in range(j - 1, -1, -1):
        #
        #         missile_j = ship.l
        #
        # for j in range(len_ssm_detections-1, -1, -1):
        #     for k in range(j-1, -1, -1):
        #         missile_j = ship.ssm_detections[j]
        #         missile_k = ship.ssm_detections[k]
        #         #print(cal_distance(missile_j, missile_k))
        #         if cal_distance(missile_j, missile_k) <= 20:
        #             edge_index[0].append(j+1)
        #             edge_index[1].append(k+1)
        #             edge_index[0].append(k+1)
        #             edge_index[1].append(j+1)
        return edge_index

    def get_enemy_edge_index(self):
        edge_index = [[],[]]
        for ship in self.friendlies_fixed_list:
            for j in range(len(self.enemies_fixed_list)):
                enemy= self.enemies_fixed_list[j]
                if enemy.status != 'destroyed':
                    edge_index[0].append(0)
                    edge_index[1].append(j+1)
        return edge_index

    def get_enemy_node_feature(self, rad_coordinate = True):
        node_features = [[0, 0, 0, 0, 0]]
        for ship in self.friendlies_fixed_list:
            for enemy in self.enemies_fixed_list:
                if enemy.status != 'destroyed':
                    if rad_coordinate == True:
                        r = ((enemy.position_x - ship.position_x) ** 2 + (enemy.position_y - ship.position_y) ** 2) ** 0.5 / (enemy.attack_range - ship.detection_range)
                        v = ((enemy.v_x - ship.v_x) ** 2 + (enemy.v_y - ship.v_y) ** 2) ** 0.5 / (self.missile_speed_scaler - ship.speed)
                        theta_r = math.atan2(enemy.position_y - ship.position_y, enemy.position_x - ship.position_x)
                        theta_v = math.atan2(ship.v_y - enemy.v_y, ship.v_x - enemy.v_x)
                        a = ((enemy.a_x - ship.a_x) ** 2 + (enemy.a_y - ship.a_y) ** 2) ** 0.5
                        theta_a = math.atan2(ship.a_y - enemy.a_y, ship.a_x - enemy.a_x)
                        if a <= 0.01:
                            a = 0
                            theta_a = 0
                        node_features.append([r, v, a, theta_r - theta_v, theta_v - theta_a])
                    else:
                        px = enemy.position_x - ship.position_x
                        py = enemy.position_y - ship.position_y
                        vx = enemy.v_x - ship.v_x
                        vy = enemy.v_y - ship.v_y
                        ax = (enemy.a_x - ship.a_x)*10
                        ay = (enemy.a_y - ship.a_y)*10
                        node_features.append([px / enemy.attack_range, py / enemy.attack_range, vx / enemy.speed, vy / enemy.speed, ax, ay])
        if ship.surface_tracking_limit+1-len(node_features)>0:
            for _ in range(ship.surface_tracking_limit+1-len(node_features)):
                node_features.append([0, 0, 0, 0, 0])
        return node_features

    def get_feature(self, ship, target):
        r = ((target.position_x - ship.position_x) ** 2 + (target.position_y - ship.position_y) ** 2) ** 0.5 / 600

        #print(ship.cla, target.cla)


        v = ((target.v_x - ship.v_x) ** 2 + (target.v_y - ship.v_y) ** 2) ** 0.5 / (self.missile_speed_scaler)

        theta_r = math.atan2(target.position_y - ship.position_y, target.position_x - ship.position_x)
        theta_v = math.atan2(ship.v_y - target.v_y, ship.v_x - target.v_x)

        a = ((target.a_x - ship.a_x) ** 2 + (target.a_y - ship.a_y) ** 2) ** 0.5
        theta_a = math.atan2(ship.a_y - target.a_y, ship.a_x - target.a_x)

        if a <= 0.01:
            a = 0
            theta_a = 0

        return r, v, a, theta_v, theta_r - theta_v, theta_v - theta_a

    def get_action_feature(self):
        dummy = [0,0,0,0,0,0,0,0]
        node_features = [dummy]
        for ship in self.friendlies_fixed_list:
            for enemy in self.enemies_fixed_list:
                if enemy.status != 'destroyed':
                    f1, f2, f3, f4, f5, f6 = self.get_feature(ship, enemy)
                    node_features.append([f1, f2, f3, f4, f5, f6, 0, 1])
                    enemy.last_action_feature = [f1, f2, f3, f4, f5, f6, 0, 1]
        if ship.surface_tracking_limit+1-len(node_features)>0:
            for _ in range(ship.surface_tracking_limit+1-len(node_features)):
                node_features.append(dummy)

        for ship in self.friendlies_fixed_list:
            for missile in ship.ssm_detections:
                f1, f2, f3, f4, f5, f6 = self.get_feature(ship, missile)
                node_features.append([f1, f2, f3, f4, f5, f6, 1, 0])
                missile.last_action_feature = [f1, f2, f3, f4, f5, f6, 1, 0]
        #print("전",len(node_features))
        if ship.surface_tracking_limit+ship.air_tracking_limit+1-len(node_features) >0:
            for _ in range(ship.surface_tracking_limit+ship.air_tracking_limit+1-len(node_features)):
                node_features.append(dummy)
        #print("후", len(node_features))

        return node_features

    def get_missile_node_feature(self, rad_coordinate = True):
        dummy =[0,
                0,
                0,
                0,
                0,
                0]
        node_features = [dummy]
        #print("node_길이 1", len(node_features))
        for ship in self.friendlies_fixed_list:
            for missile in ship.ssm_detections:
                if rad_coordinate == True:
                    f1, f2, f3, f4, f5, f6 = self.get_feature(ship, missile)
                    node_features.append([f1, f2, f3, f4, f5, f6])
                else:
                    px = missile.position_x - ship.position_x
                    py = missile.position_y - ship.position_y
                    vx = missile.v_x - ship.v_x
                    vy = missile.v_y - ship.v_y
                    ax = missile.a_x - ship.a_x
                    ay = missile.a_y - ship.a_y
                    node_features.append([px/missile.attack_range, py/missile.attack_range, vx/missile.speed, vy/missile.speed, ax, ay])
        #print("ssm detection 추가", len(node_features))
        # if ship.air_tracking_limit+1-len(node_features) > 0:
        #     for _ in range(ship.air_tracking_limit+1-len(node_features)):
        #         node_features.append(dummy)
        # print("ssm detection 추가", len(node_features))
        len_flying_sams_friendly = len(self.flying_sams_friendly)
        for j in range(len_flying_sams_friendly):
            missile = self.flying_sams_friendly[j]
            original_target = missile.original_target
            f1, f2, f3, f4, f5, f6 = self.get_feature(original_target, missile)
            node_features.append([f1, f2, f3, f4, f5, f6])
        #print("flying sam 추가", len(node_features))


        if ship.air_tracking_limit +ship.air_engagement_limit+ship.num_m_sam+1-len(node_features) > 0:
            for _ in range(ship.air_tracking_limit +ship.air_engagement_limit+ship.num_m_sam+1-len(node_features)):
                node_features.append(dummy)

        #print("마지막", len(node_features))


        #print("후", len(node_features), ship.air_tracking_limit + 1 - len(node_features))

        return node_features
                #print([px/missile.attack_range, py/missile.attack_range, vx/missile.speed, vy/missile.speed, ax/5, ay/5])




    def step(self, action_blue, action_yellow, rl = True, pass_transition = False):

        self.f11_deque.append(action_blue)
        #print(np.concatenate(list(self.f11_deque)).shape)
        self.f11 = action_blue

        #print(self.f11.shape)
        if self.visualize == True:
            self.screen.fill(self.black)

        temp_decoys_friendly = self.decoys_friendly[:]
        for decoy in temp_decoys_friendly:
            decoy.rcs_decay()
            if self.visualize == True:
                decoy.show()

        temp_decoys_enemy = self.decoys_enemy[:]
        for decoy in temp_decoys_enemy:
            decoy.rcs_decay()
            if self.visualize == True:
                decoy.show()

        for i in range(len(self.friendlies_fixed_patrol_aircraft_list)):
            patrol_aircraft = self.friendlies_fixed_patrol_aircraft_list[i]
            patrol_aircraft.maneuvering()
            if self.visualize == True:
                patrol_aircraft.show()

        for i in range(len(self.enemies_fixed_patrol_aircraft_list)):
            patrol_aircraft = self.enemies_fixed_patrol_aircraft_list[i]
            patrol_aircraft.maneuvering()
            if self.visualize == True:
                patrol_aircraft.show()

        num_f = 0
  #      reward = 0
        for i in range(len(self.friendlies_fixed_list)):
            num_f += 1
            ship = self.friendlies_fixed_list[i]

            self.temp_max_air_engagement.append(len(ship.air_engagement_managing_list))
            if ship.status != 'destroyed':
                ship.target_allot_by_action_feature(action_blue)
                ship.air_prelaunching_process()
                ship.surface_prelaunching_process()
                ship.maneuvering()
                ship.launch_decoy()
                ship.get_flying_ssms_status()
                if self.visualize == True:
                    ship.show()
                    font1 = pygame.font.Font(None, 15)
                    img1 = font1.render('SSM : {}_LSAM : {}_MSAM : {}'.format(len(ship.ssm_launcher), len(ship.l_sam_launcher), len(ship.m_sam_launcher)), True, (150, 120, 15))
                    self.screen.blit(img1, (ship.position_x, ship.position_y+20))

        for i in range(len(self.enemies_fixed_list)):
            ship = self.enemies_fixed_list[i]
            for ssm in ship.debug_ssm_launcher:
                if (ssm.status == 'flying') or (ssm.status == 'destroyed'):
                    self.debug_monitors.append(
                        ["debug", ssm.id, ssm.status, self.now, cal_distance(ssm, ssm.target), ssm.target.cla,
                         ssm.target.status])
            if ship.status != 'destroyed':
                ship.target_allocation_process(action_yellow[i])
                ship.air_prelaunching_process()
                ship.surface_prelaunching_process()
                ship.maneuvering()
                ship.launch_decoy()
                ship.get_flying_ssms_status()
                if self.visualize == True:
                    ship.show()
                    font1 = pygame.font.Font(None, 15)
                    img1 = font1.render('SSM : {}_LSAM : {}_MSAM : {}'.format(len(ship.ssm_launcher), len(ship.l_sam_launcher), len(ship.m_sam_launcher)), True, (150, 120, 15))
                    self.screen.blit(img1, (ship.position_x, ship.position_y+20))


        for i in range(len(self.friendlies_fixed_list)):
            ship = self.friendlies_fixed_list[i]
            if ship.status != 'destroyed':
                if ship.CIWS.target != None:
                    ship.CIWS.counter_attack()
                    if self.visualize == True:
                        ship.CIWS.show()

        for i in range(len(self.enemies_fixed_list)):
            ship = self.enemies_fixed_list[i]
            if ship.status != 'destroyed':
                if ship.CIWS.target != None:
                    ship.CIWS.counter_attack()
                    if self.visualize == True:
                        ship.CIWS.show()


        temp_flying_ssms_friendly = self.flying_ssms_friendly[:]
        for ssm in temp_flying_ssms_friendly:
            ssm.destroying(self.enemies,
                           self.friendlies,
                           self.flying_ssms_friendly,
                           self.flying_ssms_enemy,
                           self.flying_sams_friendly,
                           self.flying_sams_enemy)
            ssm.flying()
            ssm.seeker_operation()
            ssm.rotate_arc_beam_angle()
            if self.visualize == True:
                ssm.show()
                font1 = pygame.font.Font(None, 15)
                img1 = font1.render('{}, {}'.format(ssm.id, np.round(cal_distance(ssm, ssm.target))), True, (150, 120, 15))
                self.screen.blit(img1, (ssm.position_x, ssm.position_y))

                img1 = font1.render('est, {}, {}'.format(ssm.id, ssm.fly_mode), True, (0,250,0))
                self.screen.blit(img1, (ssm.estimated_hitting_point_x, ssm.estimated_hitting_point_y))

        temp_flying_ssms_enemy = self.flying_ssms_enemy[:]
        for ssm in temp_flying_ssms_enemy:
            ssm.destroying(self.friendlies, self.enemies, self.flying_ssms_enemy, self.flying_ssms_friendly, self.flying_sams_enemy, self.flying_sams_friendly)
            ssm.flying()
            ssm.seeker_operation()

            ssm.rotate_arc_beam_angle()
            if self.visualize == True:
                ssm.show()
                font1 = pygame.font.Font(None, 15)
                img1 = font1.render('{} ,{}'.format(ssm.id, np.round(cal_distance(ssm, ssm.target))), True, (150, 120, 15))
                self.screen.blit(img1, (ssm.position_x, ssm.position_y))

                #
                img1 = font1.render('est, {}, {}'.format(ssm.id, ssm.fly_mode), True, (0,250,0))
                self.screen.blit(img1, (ssm.estimated_hitting_point_x, ssm.estimated_hitting_point_y))

        temp_flying_sams_friendly = self.flying_sams_friendly[:]
        for sam in temp_flying_sams_friendly:
            sam.destroying(self.enemies, self.friendlies, self.flying_ssms_friendly, self.flying_ssms_enemy, self.flying_sams_friendly, self.flying_sams_enemy)
            sam.flying()
            sam.seeker_operation()
            sam.rotate_arc_beam_angle()


            if self.visualize == True:
                sam.show()
                font1 = pygame.font.Font(None, 15)
                img1 = font1.render('{}, {}, {}'.format(np.round(cal_distance(sam, sam.target)), np.round(sam.cal_distance_estimated_hitting_point()), sam.fly_mode), True, (150, 120, 15))
                self.screen.blit(img1, (sam.position_x, sam.position_y))
                img1 = font1.render('est, {}, {}'.format(sam.id, sam.fly_mode), True, (0,250,0))
                self.screen.blit(img1, (sam.estimated_hitting_point_x, sam.estimated_hitting_point_y))

        temp_flying_sams_enemy = self.flying_sams_enemy[:]
        for sam in temp_flying_sams_enemy:
            sam.destroying(self.friendlies, self.enemies, self.flying_ssms_enemy, self.flying_ssms_friendly, self.flying_sams_enemy, self.flying_sams_friendly)
            sam.flying()
            sam.seeker_operation()
            sam.rotate_arc_beam_angle()
            if self.visualize == True:
                sam.show()
                font1 = pygame.font.Font(None, 15)
                img1 = font1.render('{}, {}'.format(sam.id, sam.fly_mode), True, (0,250,0))
                self.screen.blit(img1, (sam.position_x, sam.position_y))
                #
                img1 = font1.render('est, {}, {}'.format(sam.id, sam.fly_mode), True, (0,250,0))
                self.screen.blit(img1, (sam.estimated_hitting_point_x, sam.estimated_hitting_point_y))

        if self.visualize == True:
            font1 = pygame.font.Font(None, 30)
            img1 = font1.render('{}'.format(self.now), True, (0,250,0))
            self.screen.blit(img1, (500, 500))
            self.clock.tick(self.tick)
            self.pygame.display.flip()
        self.now += self.simtime_per_framerate
        #print(self.now)

        suceptibility = None
        win_tag = None
        done = False
        #if self.now <= 2000:

        if pass_transition == False:
            missile_destroyed_cal = 0
            enemy_destroyed_cal = 0
            ship_destroyed_cal = 0
            reward=0
            for i in range(len(self.friendlies_fixed_list)):
                ship = self.friendlies_fixed_list[i]
                if ship.status != 'destroyed':pass
                    #reward += (ship.health - ship.last_health)*10
                    #ship.last_health = deepcopy(ship.health)
                    #print(reward, ship.health, ship.last_health, self.now)
                else:
                    ship_destroyed_cal += 1


            for i in range(len(self.enemies_fixed_list)):
                ship = self.enemies_fixed_list[i]
                missile_destroyed_cal += sum([1 for ssm in ship.debug_ssm_launcher if ssm.status == 'destroyed'])
                if ship.status != 'destroyed':pass
                else:
                    enemy_destroyed_cal += 1
            self.f7 = missile_destroyed_cal
            self.f8 = enemy_destroyed_cal
            self.f9 = missile_destroyed_cal - self.last_destroyed_missile
            self.f10 = enemy_destroyed_cal - self.last_destroyed_enemy
            #reward = self.
            reward = 2000 * (enemy_destroyed_cal - self.last_destroyed_enemy) \
                     -6000 * (ship_destroyed_cal - self.last_destroyed_ship) +  \
                     50 * (missile_destroyed_cal - self.last_destroyed_missile)
            reward = reward / 200
            self.last_destroyed_missile = missile_destroyed_cal
            self.last_destroyed_enemy = enemy_destroyed_cal
            self.last_destroyed_ship = ship_destroyed_cal

            if (len(self.friendlies) == 0):
                suceptibility = 1
                win_tag = "lose"
                done = True
                if self.visualize == True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            exit()

            # if (len(self.enemies) == 0):
            #     suceptibility = 0
            #     win_tag = "win"
            #     done = True
            #     if self.visualize == True:
            #         for event in pygame.event.get():
            #             if event.type == pygame.QUIT:
            #                 pygame.quit()
            #                 exit()


            if (len(self.flying_ssms_enemy) == 0) and (len(self.flying_ssms_friendly) == 0):
                done_checker_A = [True if len(enemy.ssm_launcher) == 0 else False for enemy in self.enemies]
                done_checker_B = [True if len(ship.ssm_launcher) == 0 else False for ship in self.friendlies]
                if (False in done_checker_A) or (False in done_checker_B):
                    done = False
                else:
                    #print("여긴가?", self.friendlies_fixed_list[0].status)
                    done = True

            if self.now >= 2000:
                done = True







            if self.visualize == True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

        # else:
        #     suceptibility = 0
        #     win_tag = "draw"
        #     done = True
        #     if self.visualize == True:
        #         for event in pygame.event.get():
        #             if event.type == pygame.QUIT:
        #                 pygame.quit()
        #                 exit()
        # #print(done)
        if rl == True:
            if pass_transition == False:
                return reward, win_tag, done
            else:pass
        else:
            return suceptibility, win_tag, done





