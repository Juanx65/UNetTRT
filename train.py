import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary
import numpy as np
from models.unet import U_Net
from utils.load_data import load_data
from torch.utils.data import TensorDataset, DataLoader
from utils.functions import mae_percentage


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

# Hyperparameters
num_filters = 29
val_dropout = 0.089735
learning_rate = 0.000410
batch_size = 128

# Datos de entrenamiento
x_train, x_valid, y_train, y_valid, y_mean, y_std = load_data()

x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).unsqueeze(1).float()
x_valid = torch.tensor(x_valid).float()
y_valid = torch.tensor(y_valid).unsqueeze(1).float()

tensor_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

tensor_dataset = TensorDataset(x_valid, y_valid)
valid_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

# Cargar modelo
model = U_Net(n1=num_filters, dropout_rate=val_dropout)
model.to(device)
torchsummary.summary(model, input_size=(3, 88, 32))
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento
valid_loss_min = np.Inf
n_epoch = 100
for epoch in range(n_epoch):
    train_loss = 0.0
    valid_loss = 0.0
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch) # Forward
        loss = loss_function(y_pred, y_batch) # Cálculo de la pérdida

        # Backward y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*x_batch.size(0)
    
    for batch_idx, (x_batch, y_batch) in enumerate(valid_loader):

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        loss = mae_percentage(y_pred, y_batch, y_mean, y_std)

        valid_loss +=  loss.item()*x_batch.size(0)

    if( valid_loss < valid_loss_min):
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model, 'weights/best.pth')
        valid_loss_min = valid_loss
    
    train_loss = train_loss / len(x_train)
    valid_loss = valid_loss / len(x_valid)
    if (epoch+1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 1, n_epoch, train_loss,valid_loss))


""" # Prueba del modelo entrenado
with torch.no_grad():
    test_data = torch.tensor([[5.0]])
    predicted = model(test_data)
    print(f'Predicción después de entrenamiento: f(5) = {predicted.item()}') """
