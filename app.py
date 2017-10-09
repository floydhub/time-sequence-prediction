"""
Flask Serving
This file is a sample flask app that can be used to test your model with an REST API.
This app does the following:
    - Look for sine wave test set
    - Returns the output file generated at /output


POST req:
    parameter:
        - file, required, sine waves test set

"""
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
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['pth', 'pt'])

MODEL_PATH = '/input'
print('Loading model from path: %s' % MODEL_PATH)
OUTPUT_PATH = "/output/generated.png"
# CUDA?
CUDA = torch.cuda.is_available()

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


app = Flask('Time-Sequence-Prediction')
# Return an Image
@app.route('/<path:path>', methods=['POST'])
def geneator_handler(path):
    zvector = None
    # check if the post request has the file part
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']
    if file.filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(file.filename):
        return BadRequest("Invalid file type")
    filename = secure_filename(file.filename)
    input_filepath = os.path.join('/output', filename)
    file.save(input_filepath)
    # Load a test set
    data = torch.load(input_filepath)
    # 3 samples for the test set
    input = Variable(torch.from_numpy(data[:3, :-1]), requires_grad=False)
    target = Variable(torch.from_numpy(data[:3, 1:]), requires_grad=False)
    if CUDA:
        input, target = input.cuda(), target.cuda()

    # build the model
    seq = Sequence()
    seq.double()

    ckp_name = request.form.get("ckp") or "sine_waves_lstm_model_8_epochs.pth"
    checkpoint = os.path.join(MODEL_PATH, ckp_name)
    # Load checkpoint
    if checkpoint != '':
        if CUDA:
            seq.load_state_dict(torch.load(checkpoint))
        else:
            # Load GPU model on CPU
            seq.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
    else:
        return BadRequest("Have you mount a checkpoint file for your model?")
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
    plt.savefig(OUTPUT_PATH)
    plt.close()
    return send_file(OUTPUT_PATH, mimetype='image/png')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0')
