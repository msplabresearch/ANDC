import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import eval
from pathlib import Path

#Dataset Loading
# print('Loading Dataset ...')

seeds = [45]#, 42, 10, 34, 23, 56, 31, 7, 97, 35, 2, 71, 93, 69, 53, 33, 22, 8, 19, 99]

# seeds = [99]# 35 45 8 71 22 ---- 9, 23, 56
# seed = 56
# lrs = [.700e-3, .750e-3, .725e-3, .9e-3, 1e-3]
# lrs = [.700e-3, .725e-3, 1e-3]
lr = .95e-3
for seed in seeds: 

    parser = argparse.ArgumentParser(description='AuxFormer MSP@UTD')
    parser.add_argument('-f', default='', type=str)

    parser.add_argument('--model', type=str, default='SSL_MODEL',
                        help='name of the model')
    parser.add_argument('--root', type=str, default='SSL_MODEL',
                        help='location of the root dir containing the audios/json files')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='number of classes to predict')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.2,
                        help='output layer dropout (default: 0.2')
    parser.add_argument('--layers', type=int, default = 5,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--num_heads', type=int, default = 10,
                        help='number of heads for the transformer network (default: 10)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')
    parser.add_argument('--batch_size', type = int, default = 32,
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type = float, default = 0.8,
                        help='gradient clip value (default: 0.8)')
    # parser.add_argument('--lr', type = float, default = .700e-3,
    #                     help='initial learning rate (default: .725e-3)')
    parser.add_argument('--lr', type = float, default = lr,
                        help='initial learning rate (default: .725e-3)')
    parser.add_argument('--optim', type = str, default = 'Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default = 10,
                        help='Number of Epochs (default: 20)')
    parser.add_argument('--decay', type = int, default = 5,
                        help='When to decay learning rate (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda')
    parser.add_argument('--name', type=str, default='saud_finetunrelax_v2_5_aux2_'+str(seed),
                        help='name of the saved model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = False          

    g_cuda = torch.Generator(device='cuda')

    args.a_dim, args.v_dim = 1000, 1000
    args.a_len, args.v_len = 1000, 1000
    args.use_cuda = use_cuda
    args.when = args.decay

    args.n_train = 1000
    args.n_valid =1000
    args.n_test = 1000
    args.model = str.upper(args.model.strip())
    args.output_dim = args.num_classes 
    args.criterion = 'CrossEntropyLoss'

    if __name__ == '__main__':
        file_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(file_dir, 'saved_models')
        test_loss = eval.initiate(args, model_dir)#, train_set, develop_set, test_set)
