import argparse
# vessl_on
# map_name1 = '6h_vs_8z'
# GNN = 'GAT'
def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vessl", type=bool, default=False, help="vessl AI 사용여부")
    parser.add_argument("--simtime_per_frame", type=int, default=2, help="framerate 관련")
    parser.add_argument("--decision_timestep", type=int, default=8, help="decision timestep 관련")
    parser.add_argument("--ciws_threshold", type=float, default=1, help="ciws threshold")
    parser.add_argument("--per_beta", type=float, default=0.4, help="PER_beta")
    parser.add_argument("--n_step", type=int, default=5, help="n_step")
    parser.add_argument("--vdn", type=bool, default=True, help="vdn")
    parser.add_argument("--map_name", type=str, default='6h_vs_8z', help="map name")
    parser.add_argument("--GNN", type=str, default='GAT', help="map name")
    parser.add_argument("--hidden_size_comm", type=int, default=64, help="GNN hidden layer")
    parser.add_argument("--hidden_size_enemy", type=int, default=72, help="GNN hidden layer")
    parser.add_argument("--iqn_layers", type=str, default= '[128, 96, 64, 56, 32]', help="layer 구조")
    parser.add_argument("--ship_layers", type=str, default='[64,48]', help="layer 구조")
    parser.add_argument("--missile_layers", type=str, default='[32, 32]', help="layer 구조")
    parser.add_argument("--enemy_layers", type=str, default='[32, 32]', help="layer 구조")
    parser.add_argument("--n_representation_ship", type=int, default=32, help="")
    parser.add_argument("--n_representation_missile", type=int, default=15, help="")
    parser.add_argument("--n_representation_enemy", type=int, default=15, help="")
    parser.add_argument("--iqn_layer_size", type=int, default=128, help="")
    parser.add_argument("--iqn_N", type=int, default=12, help="")
    parser.add_argument("--n_cos", type=int, default=64, help="")
    parser.add_argument("--buffer_size", type=int, default=50000, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--teleport_probability", type=float, default=0.95, help="teleport_probability")
    parser.add_argument("--gtn_beta", type=float, default=0.05, help="teleport_probability")
    parser.add_argument("--gamma", type=float, default=.99 , help="discount ratio")
    parser.add_argument("--lr", type=float, default=4e-5, help="learning rate")
    parser.add_argument("--n_multi_head", type=int, default=1, help="number of multi head")
    parser.add_argument("--num_episode", type=int, default=1000000, help="number of episode")
    parser.add_argument("--train_start", type=int, default=1000, help="number of train start")


    return parser.parse_args()
