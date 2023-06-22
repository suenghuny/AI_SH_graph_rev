import argparse
# vessl_on
# map_name1 = '6h_vs_8z'
# GNN = 'GAT'
def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vessl", type=bool, default=False, help="vessl AI 사용여부")
    parser.add_argument("--simtime_per_frame", type=int, default=2, help="framerate 관련")
    parser.add_argument("--decision_timestep", type=int, default=4, help="decision timestep 관련")
    parser.add_argument("--ciws_threshold", type=float, default=1, help="ciws threshold")
    parser.add_argument("--per_alpha", type=float, default=0.5, help="PER_alpha")
    parser.add_argument("--per_beta", type=float, default=0.6, help="PER_beta")
    parser.add_argument("--sigma_init", type=float, default=1, help="sigma_init")
    parser.add_argument("--n_step", type=int, default=5, help="n_step")
    parser.add_argument("--anneal_episode", type=int, default=1500, help="episode")
    parser.add_argument("--vdn", type=bool, default=True, help="vdn")
    parser.add_argument("--map_name", type=str, default='6h_vs_8z', help="map name")
    parser.add_argument("--GNN", type=str, default='FastGTN', help="map name")
    parser.add_argument("--hidden_size_comm", type=int, default=56, help="GNN hidden layer")
    parser.add_argument("--hidden_size_enemy", type=int, default=64, help="GNN hidden layer")
    parser.add_argument("--hidden_size_meta_path", type=int, default=56, help="GNN hidden layer")
    parser.add_argument("--iqn_layers", type=str, default= '[128,64,48,39,16]', help="layer 구조")
    parser.add_argument("--ppo_layers", type=str, default='[128,64,48,39,32]', help="layer 구조")
    parser.add_argument("--ship_layers", type=str, default='[72,56]', help="layer 구조")
    parser.add_argument("--missile_layers", type=str, default='[36,23]', help="layer 구조")
    parser.add_argument("--enemy_layers", type=str, default='[45,32]', help="layer 구조")
    parser.add_argument("--action_layers", type=str, default='[48,32]', help="layer 구조")
    parser.add_argument("--n_representation_ship", type=int, default=52, help="")
    parser.add_argument("--n_representation_missile", type=int, default=14, help="")
    parser.add_argument("--n_representation_enemy", type=int, default=28, help="")
    parser.add_argument("--n_representation_action", type=int, default=14, help="")
    parser.add_argument("--iqn_layer_size", type=int, default=64, help="")
    parser.add_argument("--iqn_N", type=int, default=48, help="")
    parser.add_argument("--n_cos", type=int, default=36, help="")
    parser.add_argument("--buffer_size", type=int, default=50000, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--teleport_probability", type=float, default=1.0, help="teleport_probability")
    parser.add_argument("--gtn_beta", type=float, default=0.05, help="teleport_probability")
    parser.add_argument("--gamma", type=float, default=.99, help="discount ratio")
    parser.add_argument("--lr", type=float, default=0.9e-4, help="learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_multi_head", type=int, default=1, help="number of multi head")
    parser.add_argument("--num_episode", type=int, default=1000000, help="number of episode")
    parser.add_argument("--scheduler_step", type =int, default=100, help= "scheduler step")
    parser.add_argument("--scheduler_ratio", type=float, default=0.992, help= "scheduler ratio")
    parser.add_argument("--train_start", type=int, default=1, help="number of train start")
    parser.add_argument("--epsilon_greedy", type=bool, default=True, help="epsilon_greedy")
    parser.add_argument("--epsilon", type=float, default=0.5, help="epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.01, help="epsilon")
    parser.add_argument("--anneal_step", type=int, default=50000, help="epsilon")
    parser.add_argument("--temperature", type=int, default=7, help="")
    parser.add_argument("--interval_min_blue", type=bool, default=True, help="interval_min_blue")
    parser.add_argument("--interval_constant_blue", type=float, default=4, help="interval_constant_blue")
    parser.add_argument("--action_history_step", type=int, default=4, help="action_history_step")
    parser.add_argument("--graph_distance", type=float, default=20, help="graph distance")
    parser.add_argument("--grad_clip", type=float, default=0.5, help="gradient clipping")
    parser.add_argument("--test_epi", type=int, default=1800, help="interval_constant_blue")
    parser.add_argument("--scheduler", type=str, default='step', help="step 형태")
    parser.add_argument("--t_max", type=int, default=40000, help="interval_constant_blue")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE lmbda")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="clipping epsilon")
    parser.add_argument("--K_epoch", type=int, default=3, help="K-epoch")
    parser.add_argument("--num_GT_layers", type=int, default=2, help="num GT layers")
    parser.add_argument("--channels", type=int, default=1, help="channels")
    parser.add_argument("--num_layers", type=int, default=2, help="num layers")
    parser.add_argument("--embedding_train_stop", type=int, default=100, help="embedding_train_stop")
    parser.add_argument("--n_eval", type=int, default=1, help="number of evaluation")
    parser.add_argument("--with_noise", type=bool, default=False, help="")
    parser.add_argument("--temp_constant", type=float, default=1, help="")
    parser.add_argument("--init_constant", type=int, default=10000, help="")
    parser.add_argument("--cuda", type=str, default='cuda:0', help="")
    return parser.parse_args()