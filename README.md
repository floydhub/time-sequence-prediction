# Time Sequence Prediction
This is a toy example for beginners to start with, more in detail: it's a porting of [pytorch/examples/time-sequence-prediction](https://github.com/pytorch/examples/tree/master/time_sequence_prediction) making it usables on [FloydHub](https://www.floydhub.com/). It is helpful for learning both pytorch and time sequence prediction. Two [LSTMCell](http://pytorch.org/docs/master/nn.html?highlight=lstmcell#torch.nn.LSTMCell) units are used in this example to learn some sine wave signals starting at different phases. After learning the sine waves, the network tries to predict the signal values in the future. The results is shown in the picture below.

![image](https://cloud.githubusercontent.com/assets/1419566/24184438/e24f5280-0f08-11e7-8f8b-4d972b527a81.png)

The initial signal and the predicted results are shown in the image. We first give some initial signals (full line). The network will  subsequently give some predicted results (dash line). It can be concluded that the network can generate new sine waves.


## Usage

```bash
# Generate Sine wave dataset
$ python generate_sine_wave.py
# Train a 2 LSTM cell model
$ python train.py

# Generate Sine wave testset
$ python generate_sine_wave.py --test
# Evaluate
$ python eval.py --ckp <CHECKPOINT_FILE>
```


The `generate_sine_wave.py` script accepts the following arguments:

```bash
usage: generate_sine_wave.py [-h] [--outf OUTF] [--nsamples N] [--lenght L]
                             [--test]

Sine Waves Generator

optional arguments:
  -h, --help    show this help message and exit
  --outf OUTF   folder to output generated dataset
  --nsamples N  number of samples (default: 100)
  --lenght L    lenght (default: 1000)
  --test        create a 3 samples test set
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


The `eval.py` script accepts the following arguments:

```bash
usage: eval.py [-h] [--data DATA] [--outf OUTF] [--ckpf CKPF]

Pytorch Time Sequence Prediction

optional arguments:
  -h, --help   show this help message and exit
  --data DATA  path to dataset(testset)
  --outf OUTF  folder to output images and model checkpoints
  --ckpf CKPF  path to model checkpoint file (to continue training)
```

## Architecture

![LSTM](images/LSTM3-chain.png)

*Credit: [colah.github.io](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)*

Note: There are 2 differences from the image above with respect the model used in this example:

- This model use 2 `LSTMCell`,
- The output of first LSTM is used as input for the second LSTM cell.

## Run on FloydHub

Here's the commands to training, evaluating and serving your time sequence prediction model on FloydHub.

### Project Setup

Before you start, log in on FloydHub with the [floyd login](http://docs.floydhub.com/commands/login/) command, then fork and init the project:

```bash
$ git clone https://github.com/floydhub/time-sequence-prediction.git
$ cd time-sequence-prediction
$ floyd init time-sequence-prediction
```

### Training

Before you start, run `python generate_sine_wave.py` and upload the generated dataset(`traindata.pt`) as FloydHub dataset, following the FloydHub docs: [Create and Upload a Dataset](https://docs.floydhub.com/guides/create_and_upload_dataset/). I've already uploaded a dataset for you if you want to skip this step.

Now it's time to run our training on FloydHub. In this example we will train the model for 8 epochs with a gpu instance.

```bash
floyd run --gpu --env pytorch-0.2:py2 --data redeipirati/datasets/sine-waves/1:input "python train.py"
```


Note:

- `--gpu` run your job on a FloydHub GPU instance
- `--env pytorch-0.2` prepares a pytorch environment for python 3.
- `--data redeipirati/datasets/sine-waves/1` mounts the pytorch mnist dataset in the `/input` folder inside the container for our job.

You can follow along the progress by using the [logs](http://docs.floydhub.com/commands/logs/) command. The training should take about 5 minutes on a GPU instance and about 15 minutes on a CPU one.

### Evaluating

First of all, geneated a test set running `python generate_sine_wave.py --test`, then run:

```bash
floyd run --gpu --env pytorch-0.2:py2 --data <REPLACE_WITH_JOB_OUTPUT_NAME>:model "python eval.py --ckp /model/sine_waves_lstm_model_8_epochs.pth"
```

### Serve model through REST API

FloydHub supports seving mode for demo and testing purpose. Before serving your model through REST API,
you need to create a `floyd_requirements.txt` and declare the flask requirement in it. If you run a job
with `--mode serve` flag, FloydHub will run the `app.py` file in your project
and attach it to a dynamic service endpoint:


```bash
floyd run --gpu --mode serve --env pytorch-0.2:py2 --data <REPLACE_WITH_JOB_OUTPUT_NAME>:input
```

The above command will print out a service endpoint for this job in your terminal console.

The service endpoint will take a couple minutes to become ready. Once it's up, you can interact with the model by sending sine waves file with a POST request and the service will return the predicted sequences:

```bash
# Template
curl -X POST -o <NAME_&_PATH_DOWNLOADED_IMG> -F "file=@<TEST_DATA_FILE>" -F "ckp=<MODEL_CHECKPOINT>" <SERVICE_ENDPOINT>

# e.g. of a POST req
curl -X POST -o prova.png -F "ile=@./traindata.pth" https://www..floydlabs.com/expose/BhZCFAKom6Z8RptVKskHZW
```

Any job running in serving mode will stay up until it reaches maximum runtime. So
once you are done testing, **remember to shutdown the job!**

*Note that this feature is in preview mode and is not production ready yet*

### More resources

Some useful resources on LSTM Cell and Networks:
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Understanding LSTM and its diagrams](https://medium.com/@shiyan/understanding-lstm-and-its-diagrams-37e2f46f1714)
- [Exploring LSTMs - see Snorlax Example :)](http://blog.echen.me/2017/05/30/exploring-lstms/)
- [What is an intuitive explanation of LSTMs and GRUs?](https://www.quora.com/What-is-an-intuitive-explanation-of-LSTMs-and-GRUs)
- [Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) - Brandon Rohrer](https://youtu.be/WCUNPb-5EYI)

### Contributing

For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!
