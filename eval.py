import torch
import random
import os
import numpy as np
from models.unet import U_Net
import matplotlib.pyplot as plt
from utils.functions import destandarize
from utils.load_data import load_test_data
from matplotlib.ticker import MaxNLocator

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Using CPU')
else:
    print('CUDA is available. Using GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

# Hyperparameters
num_filters = 29
val_dropout = 0.089735
learning_rate = 0.000410
batch_size = 128

x_test, y_test, y_mean, y_std, fs_test, r_test, z_test= load_test_data()

x_test = torch.tensor(x_test).float().to(device)
y_test = torch.tensor(y_test).float().to(device)


####LOAD######
model = U_Net(n1=num_filters, dropout_rate=val_dropout)
model = torch.load('weights/best.pth')
model.to(device)
model.eval()

# Eval 
with torch.no_grad():
        output = model(x_test)

x_test = x_test.cpu()
y_test = y_test.cpu()

y_test_pred = destandarize(output, y_mean, y_std)
y_test_pred = y_test_pred.squeeze(1)

y_test_pred = y_test_pred.cpu()

print("x_test size", x_test.size())

N = random.randint(0,len(x_test)-1)
print(N)

y_test[N][fs_test[N] <= 0.05e-6] = 0
y_test_pred[N][fs_test[N] <= 0.05e-6] = 0

abs_err = np.abs(y_test[N] - y_test_pred[N])
print('Abs. error max:', abs_err.max())
print('Abs. error min:', abs_err.min())
print('Abs. error mean:', abs_err.mean())
print('Abs. error stddev:', abs_err.std())
print('Abs. error %:', abs_err.mean()*100/y_test[N].mean())