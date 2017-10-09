import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='Sine Waves Generator')
parser.add_argument('--outf', default='./',
                    help='folder to output generated dataset')
parser.add_argument('--nsamples', type=int, default=100, metavar='N',
                    help='number of samples (default: 100)')
parser.add_argument('--lenght', type=int, default=1000, metavar='L',
                    help='lenght (default: 1000)')
parser.add_argument('--test', action='store_true',
					help='create a 3 samples test set')
args = parser.parse_args()
np.random.seed(2)

T = 20
L = args.lenght
# Create a 3 Sample testset if test is given else use nsamples
N = args.nsamples if not args.test else 3

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))
