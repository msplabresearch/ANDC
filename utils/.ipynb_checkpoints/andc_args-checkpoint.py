import argparse

def get_args():
    parser = argparse.ArgumentParser(description='ANDC')
    # parser.add_argument('-f', default='', type=str)

    parser.add_argument('-r', '--root', type=str, default='.',
                        help='The root directory containing the JSON files')

    parser.add_argument('-s','--opensmile', type=str, default='SMILExtract',
                        help='Path to the SMILExtract from opensmile')
    
    parser.add_argument("-set", "--dataset", default='MSP-Podcast',
                        help='type of model/dataset to used if applicable')
    
    parser.add_argument("-m", "--model_path", default='MSP-Podcast',
                        help='type of model/dataset to used if applicable')
    
    parser.add_argument('--audio_dir', type=str, default='./models',
                        help='The root directory containing the DL model')
    
    parser.add_argument("-batch", "--batch_size", default=1)
    
    parser.add_argument('--seed', type=str, default='42',
                        help='Seed number')
    return parser.parse_args()

# parser.add_argument('--model', type=str, default='SSL_MODEL',
#                     help='name of the model')

# parser.add_argument('--num_classes', type=int, default=6,
#                     help='number of classes to predict')

# parser.add_argument('--attn_dropout', type=float, default=0.1,
#                     help='attention dropout')

# parser.add_argument('--relu_dropout', type=float, default=0.1,
#                     help='relu dropout')

# parser.add_argument('--embed_dropout', type=float, default=0.25,
#                     help='embedding dropout')

# parser.add_argument('--res_dropout', type=float, default=0.1,
#                     help='residual block dropout')

# parser.add_argument('--out_dropout', type=float, default=0.2,
#                     help='output layer dropout (default: 0.2')

# parser.add_argument('--layers', type=int, default = 5,
#                     help='number of layers in the network (default: 5)')

# parser.add_argument('--num_heads', type=int, default = 10,
#                     help='number of heads for the transformer network (default: 10)')

# parser.add_argument('--attn_mask', action='store_false',
#                     help='use attention mask for Transformer (default: true)')

# parser.add_argument('--batch_size', type = int, default = 32,
#                     help='batch size (default: 32)')
# parser.add_argument('--seed', type=int, default=42,
#                     help='random seed')

# parser.add_argument('--no_cuda', action='store_true',
#                     help='do not use cuda')