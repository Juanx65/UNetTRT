import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary
import numpy as np
from utils.models.unet import U_Net
from utils.load_data import MyDataLoader
from torch.utils.data import TensorDataset, DataLoader
from utils.functions import mae_percentage
import argparse


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

def train(opt):
    # Datos de entrenamiento
    my_data_loader = MyDataLoader()
    x_train, x_valid, y_train, y_valid, y_mean, y_std = my_data_loader.load_data()

    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).unsqueeze(1).float()
    x_valid = torch.tensor(x_valid).float()
    y_valid = torch.tensor(y_valid).unsqueeze(1).float()

    tensor_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(tensor_dataset, batch_size=opt.batch_size, shuffle=True)

    tensor_dataset = TensorDataset(x_valid, y_valid)
    valid_loader = DataLoader(tensor_dataset, batch_size=opt.batch_size, shuffle=False)

    # Cargar modelo
    model = U_Net(n1=opt.num_filters, dropout_rate=opt.dropout)
    model.to(device)
    torchsummary.summary(model, input_size=(3, 128, 32))
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    # Entrenamiento
    valid_loss_min = np.Inf
    n_epoch = opt.epochs
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
            torch.save(model, opt.weights)
            valid_loss_min = valid_loss
        
        train_loss = train_loss / len(x_train)
        valid_loss = valid_loss / len(x_valid)
        if (epoch+1) % 1 == 0:
            print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 1, n_epoch, train_loss,valid_loss))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 32, type=int,help='batch size to train')
    parser.add_argument('--epochs', default = 100, type=int,help='epoch to train')
    parser.add_argument('--dropout', default = 0.089735, type=float,help='percentage dropout to use')
    parser.add_argument('--num_filters', default = 29, type=int,help='Canales de salida de la primera capa conv')
    parser.add_argument('--learning_rate', default = 0.000410, type=float, help='learning rate')
    parser.add_argument('--weights', default = 'weights/best.pth', type=str, help='directorio y nombre de archivo de donse se guardara el mejor peso entrenado')
    opt = parser.parse_args()
    return opt

def main(opt):
    train(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
