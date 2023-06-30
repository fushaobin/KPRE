import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="KPRE")

    parser.add_argument('--dataset', type=str, default='music',
                        help='which dataset to use (music, book, movie, restaurant)')

    parser.add_argument('--n_epoch', type=int, default=25, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--adj_size', type=int, default=8, help="Number of user sampled items")
    parser.add_argument('--n_layer', type=int, default=3, help='depth of the sampling layer')
    parser.add_argument('--aim_num', type=int, default=3, help='number of indicators sampled per layer')

    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
    parser.add_argument('--dim', type=int, default=64, help='dimension of node and relation embeddings')
    parser.add_argument('--agg', type=str, default='concat',
                        help='which aggregator to use (average, max, min, concat)')

    parser.add_argument("--device", type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
    parser.add_argument('--random_flag', type=bool, default=True, help='whether using random seed or not')

    return parser.parse_args()