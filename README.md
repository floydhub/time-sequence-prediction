# Time Sequence Prediction
This is a toy example for beginners to start with, more in detail: it's a porting of [pytorch/examples/mnist](https://github.com/pytorch/examples/tree/master/time_sequence_prediction) making it usables on [FloydHub](https://www.floydhub.com/). It is helpful for learning both pytorch and time sequence prediction. Two [LSTMCell]() units are used in this example to learn some sine wave signals starting at different phases. After learning the sine waves, the network tries to predict the signal values in the future. The results is shown in the picture below.

![image](https://cloud.githubusercontent.com/assets/1419566/24184438/e24f5280-0f08-11e7-8f8b-4d972b527a81.png)

The initial signal and the predicted results are shown in the image. We first give some initial signals (full line). The network will  subsequently give some predicted results (dash line). It can be concluded that the network can generate new sine waves.


## Usage

```bash
# Generate Sine wave dataset
python generate_sine_wave.py
# Train a 2 LSTM cell model
python train.py
```


The `train.py` script accepts the following arguments:

```bash
usage: train.py [-h] [--data DATA] [--lr LR] [--outf OUTF] [--epochs N]

Pytorch Time Sequence Prediction

optional arguments:
  -h, --help   show this help message and exit
  --data DATA  path to dataset
  --lr LR      learning rate (default: 0.01)
  --outf OUTF  folder to output images and model checkpoints
  --epochs N   number of epochs to train (default: 8)

```


## Architecture

## Run on FloydHub

Here's the commands to training, evaluating and serving your time sequence prediction model on FloydHub.

### Project Setup

Before you start, log in on FloydHub with the [floyd login](http://docs.floydhub.com/commands/login/) command, then fork and init the project:

```bash
$ git clone https://github.com/ReDeiPirati/time-sequence-prediction.git
$ cd time-sequence-prediction
$ floyd init time-sequence-prediction
```

### Training

Before you start, run `python generate_sine_wave.py` and upload the generated dataset(`traindata.pt`) as FloydHub dataset, following the FloydHub docs: [Create and Upload a Dataset](https://docs.floydhub.com/guides/create_and_upload_dataset/).

Now it's time to run our training on FloydHub. In this example we will train the model for 8 epochs with a gpu instance.

```bash
floyd run --gpu --env pytorch-0.2  --data redeipirati/datasets/sine-waves/1:input "python train.py"
```

### Evaluating

### Serve model through REST API

### More resources

Some useful resources on LSTM Cell and Networks:
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Understanding LSTM and its diagrams](https://medium.com/@shiyan/understanding-lstm-and-its-diagrams-37e2f46f1714)
- [Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) - Brandon Rohrer](https://youtu.be/WCUNPb-5EYI)

### Contributing

For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!
