import argparse
import torch
import random
import numpy as np

_DEVICE_DEFAULT = 'cuda' if torch.cuda.is_available() else 'cpu'
_ARGS = None

def get_parser():
    parser = argparse.ArgumentParser(description='GEP MAGEP Training')

    parser.add_argument('--geno_dir',   type=str, default='./test_data/case3/genotype',help='genotype data')
    parser.add_argument('--env_dir',    type=str, default='./test_data/case3/environment',help='environment data')
    parser.add_argument('--pheno_dir',  type=str, default='./test_data/case3/phenotype', help='phenotype')


    parser.add_argument('--source_locations', type=str, default='GAH1,GAH2,ARH1,ARH2,TXH1,TXH2', help='source domain')
    parser.add_argument('--target_locations', type=str, default='IAH1,IAH2,IAH3,IAH4', help='target domain')


    parser.add_argument('--seed',type=int, default=42, help='SEED')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--support_ratio', type=float, default=0.2, help='Ratio of support set')
    parser.add_argument('--device', type=str, default=_DEVICE_DEFAULT, help="'cuda' OR 'cpu'")

    parser.add_argument('--save_dir', type=str, default='./save_model/meta-da-coral', help='Save model')
    parser.add_argument('--pic_dir',  type=str, default='./save_result/meta-da-coral', help='Save result')

    parser.add_argument('--meta_lr',       type=float, default=0.001,  help='source domain meta learning rate')
    parser.add_argument('--inner_lr',      type=float, default=5e-4,   help='source domain internal loop learning rate')
    parser.add_argument('--inner_updates', type=int,   default=20,     help='source domain internal loop update steps')
    parser.add_argument('--episodes',      type=int,   default=200,    help='source domain number of episodes')
    parser.add_argument('--k_tasks',       type=int,   default=4,      help='Number of sampling tasks per round')


    parser.add_argument('--lr',  type=float, default=8e-4,  help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,  help='weight decay')
    parser.add_argument('--epochs',  type=int,   default=200,    help='epochs')
    parser.add_argument('--early_stop_patience', type=int,  default=30,  help='patience')
    parser.add_argument('--scheduler_factor',   type=float, default=0.3,  help='scheduler factor')
    parser.add_argument('--scheduler_patience',  type=int,   default=15,   help='scheduler factor')

    return parser

def parse_args(argv=None):
    return get_parser().parse_args(argv)

def init_args(args=None):
    global _ARGS
    if args is None:
        _ARGS = parse_args()
    else:
        _ARGS = args
    return _ARGS
def get_args():
    if _ARGS is None:
        raise RuntimeError('args has not been initialized yet')
    return _ARGS

def apply_seed(seed=None):
    if seed is None:
        seed = get_args().seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def device():
    return get_args().device
