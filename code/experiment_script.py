# -*- coding: utf-8 -*-
"""「optdl2023-exp-why.ipynb」的副本

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Eb1c3rrTRWppOTSUQ-1IANSWFVopcAdc
"""
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from time import time
from tqdm import tqdm
from torch.autograd.functional import jacobian
# from torch.autograd.functional import gradient
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.profiler as profiler
from time import time
import os



def experiment(args):
    """# TensorFlow
    https://www.tensorflow.org/api_docs/python/tf/GradientTape#batch_jacobian
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    batch_size = 2
    n, m = 100, 100
    num_trials = 100
    tf.config.set_visible_devices([], 'GPU')
    i_s = None
    model = tf.keras.models.Sequential()

    if args.exp == "FCN":
        ### FCN
        model.add(tf.keras.Input(shape=n))
        model.add(tf.keras.layers.Dense(m)) 
        max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')
        x = tf.random.uniform([batch_size, n])
        @tf.function
        def compute_jacobian(model, x, i_s=None):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                y = model(x)
            return tape.jacobian(y, x)

        with tf.GradientTape(persistent=True) as tape: # Eager
            tape.watch(x)
            y = model(x)
    elif args.exp == "CNN":
        ### CNN
        x = tf.random.uniform([batch_size, 8, 1]) 
        model.add(tf.keras.layers.Conv1D(1, 3, padding='valid'))


        @tf.function
        def compute_jacobian(model, x, i_s=None):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                y = model(x)
            return tape.jacobian(y, x)

        with tf.GradientTape(persistent=True) as tape: # Eager
            tape.watch(x)
            y = model(x)

    else:
        ### LSTM
        embedding_dim, state_dim = 15, 15
        model = tf.keras.layers.LSTMCell(embedding_dim)
        x = tf.random.uniform([batch_size, embedding_dim])
        i_s = (tf.Variable(tf.random.uniform([batch_size, state_dim])), tf.Variable(tf.random.uniform([batch_size, state_dim])))
        @tf.function
        def compute_jacobian(model, x, i_s=None):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                if i_s:
                    tape.watch(i_s)
                    y = model(x,i_s)[0]
                else:
                    y = model(x)
            return tape.jacobian(y, x)

        with tf.GradientTape(persistent=True) as tape: # Eager
            tape.watch(x)
            tape.watch(i_s)
            y = model(x, states=i_s)[0]

    # [ 2 100 100], shape=(3,) => [2 100 100], shape=(3,),


    run_time = []

    
    for epoch in tqdm(range(num_trials)):
        start = time()
        J_tf = compute_jacobian(model, x, i_s)
        run_time.append(time() - start)
    print(tf.shape(x), tf.shape(y), tf.shape(J_tf)) 
      

    print("\n\n---Summary---",args.exp)
    print(np.mean(run_time))
    print(np.std(run_time))

    """# PyTorch
    https://github.com/pytorch/pytorch/blob/main/torch/autograd/functional.py#L499
    """

    device = "cpu"


    ### FCN
    if args.exp == "FCN":
        x = torch.rand([batch_size, n], device="cpu", requires_grad=True) # FCN
        # Define the model
        class FullyConnected(nn.Module):
            def __init__(self, input_size, inner_size):
                super(FullyConnected, self).__init__()
                self.fc = nn.Linear(input_size, inner_size)
                self.fc2 = nn.Linear(inner_size, 1)

            def forward(self, x):
                x = self.fc2(self.fc(x))
                return x
        model = FullyConnected(n, m).to("cpu") # FCN

    ### CNN
    elif args.exp == "CNN":
        x = torch.rand([batch_size, 1, 8], device="cpu", requires_grad=True) 
        class CNN(nn.Module):
            def __init__(self, input_channel, output_channel, kernel_size):
                super(CNN, self).__init__()
                self.cnn = nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size)

            def forward(self, x):
                x = self.cnn(x)
                return x

        model = CNN(1, 1, 3).to("cpu")


    ### LSTM
    else:
        embedding_dim, state_dim = 15, 15
        x = torch.rand([batch_size, embedding_dim], device="cpu", requires_grad=True) # batch, seq, feature
        class LSTM(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(LSTM, self).__init__()
                self.lstm = torch.nn.LSTMCell(input_size, hidden_size, bias=True)

            def forward(self, x):
                x = self.lstm(x)
                # [2, 10, 20], [4, 2, 10]
                return x[0]

        model = LSTM(embedding_dim, state_dim).to("cpu")

    run_time = []


    for epoch in tqdm(range(num_trials)):
        cur_time = time()
        J_pt = jacobian(model, x)
        aft_time = time()
        run_time.append(aft_time - cur_time)
    print(x.shape)
    print(J_pt[0].shape) # [10, 20, 2, 10, 15]
    print(J_pt[1].shape)


    print("\n\n---Summary---",args.exp)
    print(np.mean(run_time))
    print(np.std(run_time))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp")
    args = parser.parse_args()
    experiment(args)

"""
FCN
TensorFlow
0.00284116268157959
0.018887278824266846

PyTorch
0.000634000301361084
0.0006127831682779093

CNN
TensorFlow
0.002322978973388672
0.01757940397511298

PyTorch
0.0031055641174316407
0.0007723201148682738

LSTM
TensorFlow
0.0040093731880187986
0.030208463640201286

PyTorch
0.005568251609802246
0.0022453100052345428
"""
