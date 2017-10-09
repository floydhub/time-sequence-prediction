from __future__ import print_function
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Pytorch Time Sequence Prediction')
parser.add_argument('--data', default="/input/" ,help='path to dataset')
parser.add_argument('--lr', type=float, default=0.4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--outf', default='/output',
                    help='folder to output images and model checkpoints')
parser.add_argument('--epochs', type=int, default=8, metavar='N',
                    help='number of epochs to train (default: 8)')

args = parser.parse_args()
# CUDA?
CUDA = torch.cuda.is_available()

# Is there the outf?
try:
    os.makedirs(args.outf)
except OSError:
    pass

# Model
class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 1)
        if CUDA:
            self.lstm1, self.lstm2 = self.lstm1.cuda(), self.lstm2.cuda()

    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=False)
        if CUDA:
            h_t, c_t, h_t2, c_t2 = h_t.cuda(), c_t.cuda(), h_t2.cuda(), c_t2.cuda()

        # Iterate over columns
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            outputs += [h_t2]

        # Begin with the test input and continue for steps in range(future) predictions
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(h_t2, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            outputs += [h_t2]
        # Compact the list of predictions
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load(os.path.join(args.data, 'traindata.pt'))
    # Sample: [1, 2, 3, 4]
    # Input less the last value(what we want to predict given the sequence)
    # e.g. [1, 2, 3]
    input = Variable(torch.from_numpy(data[3:, :-1]), requires_grad=False)
    # Predict the next value (move the input one position right)
    # e.g. [2, 3, 4]
    target = Variable(torch.from_numpy(data[3:, 1:]), requires_grad=False)
    if CUDA:
        input, target = input.cuda(), target.cuda()

    # 3 samples for the test set
    test_input = Variable(torch.from_numpy(data[:3, :-1]), requires_grad=False)
    test_target = Variable(torch.from_numpy(data[:3, 1:]), requires_grad=False)
    if CUDA:
        test_input, test_target = test_input.cuda(), test_target.cuda()
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    if CUDA:
        criterion.cuda()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=args.lr)
    # begin to train
    for i in range(args.epochs):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.data.cpu().numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict
        future = 1000
        pred = seq(test_input, future = future)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss:', loss.data.cpu().numpy()[0])
        y = pred.data.cpu().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig(os.path.join(args.outf, 'predict%d.png'%i))
        plt.close()

    # Do checkpointing - Is saved in outf
    torch.save(seq.state_dict(), '%s/sine_waves_lstm_model_%d_epochs.pth' % (args.outf, args.epochs))
