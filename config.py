import argparse
from models import args_for_model

def args_for_data(parser):
    parser.add_argument('--path', type=str, default='../data/cartoonset100k')
    parser.add_argument('--result_path', type=str, default='./result')
    
def args_for_train(parser):
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--batch_size', type=int, default=None, help='batch_size')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stopping')    
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for the optimizer')
    parser.add_argument('--scheduler', type=str, default='None')
    parser.add_argument('--warmup_epochs', type=int, default=-1, help='number of warmup epoch of lr scheduler')

    parser.add_argument('--continue_train', type=int, default=-1, help='continue training from fold x') 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='?', type=str)

    args_for_data(parser)
    args_for_train(parser)
    _args, _ = parser.parse_known_args()
    args_for_model(parser, _args.model)

    args = parser.parse_args()
    return args
