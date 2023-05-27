import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from time import time
from tqdm import tqdm
from torch.autograd.functional import jacobian
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.profiler as profiler
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
batch_size = 1
n, m = 100, 100
num_trials = 10000

# Define the models
# TensorFlow
tf.config.set_visible_devices([], 'GPU')
tf_model = tf.keras.models.Sequential()
tf_model.add(tf.keras.Input(shape=n))
tf_model.add(tf.keras.layers.Dense(m))
# model.add(tf.keras.layers.Dense(1))

# PyTorch
class FullyConnected(nn.Module):
    def __init__(self, input_size, inner_size):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(input_size, inner_size)
        self.fc2 = nn.Linear(inner_size, 1)

    def forward(self, x):
        x = self.fc2(self.fc(x))
        return x
pt_model = FullyConnected(n, m).to("cpu")

# Create Inputs
tf_x = tf.random.uniform([batch_size, n])
pt_x = torch.rand([batch_size, n], device="cpu", requires_grad=True)
tf_run_time = []
pt_run_time = []

@tf.function
def compute_jacobian(model, x):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = model(x)
    return tape.jacobian(y, x)

for epoch in tqdm(range(num_trials)):
    start = time()
    J_tf = compute_jacobian(tf_model, tf_x)
    tf_run_time.append(time() - start)

    start = time()
    J_pt = jacobian(pt_model, pt_x)
    pt_run_time.append(time() - start)

eager_run_time = []

with tf.GradientTape(persistent=True) as tape:
    tape.watch(tf_x)
    tf_y = tf_model(tf_x)

for epoch in tqdm(range(100)):
    start = time()
    J_tf = tape.jacobian(tf_y, tf_x)
    eager_run_time.append(time() - start)

print("\n\n---Summary---")
print("TensorFlow")
print('mean', np.mean(tf_run_time))
print('std', np.std(tf_run_time))

print("PyTorch")
print('mean', np.mean(pt_run_time))
print('std', np.std(pt_run_time))

print("TensorFlow (eager)")
print('mean', np.mean(eager_run_time))
print('std', np.std(eager_run_time))