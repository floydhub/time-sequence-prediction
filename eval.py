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
parser.add_argument('--data', default="./" ,help='path to dataset(testset)')
parser.add_argument('--outf', default='/output',
					help='folder to output images and model checkpoints')
parser.add_argument('--ckpf', default='',
					help="path to model checkpoint file (to continue training)")
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
	# load data and make training set
	data = torch.load(os.path.join(args.data, 'traindata.pt'))

	# 3 samples for the test set
	input = Variable(torch.from_numpy(data[:3, :-1]), requires_grad=False)
	target = Variable(torch.from_numpy(data[:3, 1:]), requires_grad=False)
	if CUDA:
		input, target = input.cuda(), target.cuda()

	# build the model
	seq = Sequence()
	seq.double()

	# Load checkpoint
	if args.ckpf != '':
		if CUDA:
			seq.load_state_dict(torch.load(args.ckpf))
		else:
			# Load GPU model on CPU
			seq.load_state_dict(torch.load(args.ckpf, map_location=lambda storage, loc: storage))
	else:
		print ("You need to specify a checkpoint file")
		exit(-1)

	# begin to predict
	future = 1000
	pred = seq(input, future = future)
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
	plt.savefig(os.path.join(args.outf, 'eval_predict.png'))
	plt.close()
