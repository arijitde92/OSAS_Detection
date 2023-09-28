import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn import NLLLoss, BCELoss, BCEWithLogitsLoss
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchinfo import summary
from data_utils import load_data
from model import ConvNet
from Dataset import OSASUDDataset

DATA_FILE_NAME = 'osasud_numpy_processed.pkl'
DATA_DIR = 'data'
MODEL_SAVE_DIR = 'trained_models'
OUTPUT_DIR = 'output'
BATCH_SIZE = 128
LR = 0.00002
N_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot(train_losses, train_acc, test_losses, test_acc, label):
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    axs[0].plot(test_losses, label='val loss')
    axs[0].plot(train_losses, label='train loss')
    axs[0].set_title("Loss")
    axs[1].plot(test_acc, label='val accuracy')
    axs[1].plot(train_acc, label='train accuracy')
    axs[1].set_title(label)
    plt.savefig(f'{label}.png')
    plt.show()


def train(model, loader, loss_function, optimizer):
    model.train()
    pbar = tqdm(loader)
    running_loss = 0.0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = loss_function(y_pred, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f'Loss={running_loss} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
    train_loss = running_loss / len(loader.dataset)
    train_accuracy = 100 * correct / processed
    return train_loss, train_accuracy


def test(loader, model, loss_function):
    model.eval()
    pbar = tqdm(loader)
    running_loss = 0.0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(DEVICE), target.to(DEVICE)
        y_pred = model(data)
        loss = loss_function(y_pred, target)
        running_loss += loss.item()

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f'Loss={running_loss} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
    test_loss = running_loss / len(loader.dataset)
    test_accuracy = 100 * correct / processed
    return test_loss, test_accuracy


if __name__ == '__main__':
    # classification: 0 = Binary, 1 = Multiclass
    x_train, y_train, x_test, y_test = load_data(DATA_FILE_NAME, classification=0)
    if np.isnan(x_train).any() or np.isnan(x_test).any():
        print("NaN in input")
        exit(1)
    train_dataset = OSASUDDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = OSASUDDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    num_classes = len(np.unique(y_train))
    print("Number of classes:", num_classes)
    model = ConvNet(x_train.shape[1:], num_classes=num_classes)
    model.to(DEVICE)
    if num_classes > 2:
        # Multi Class classification
        loss_function = NLLLoss()
        print("Loss Function: NLL Loss")
    else:
        # Binary Classification
        loss_function = BCELoss()
        print("Loss Function: BCE Loss")

    optimizer = Adam(model.parameters(), lr=LR)
    input_size = (BATCH_SIZE, 4, 80)
    summary(model, input_size=input_size)
    epoch_train_acc = []
    epoch_train_loss = []
    epoch_valid_acc = []
    epoch_valid_loss = []
    min_val_loss = 99999
    for epoch in range(N_EPOCHS):
        print("EPOCH: %s" % epoch)
        train_loss, train_acc = train(model, train_loader, loss_function, optimizer)
        test_loss, test_acc = test(model, test_loader, loss_function)
        if test_loss < min_val_loss:
            print("Validation Loss decreased, saving model")
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(test_acc)
        epoch_valid_loss.append(test_loss)
    plot(epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc,
         f'Loss & Accuracy')
