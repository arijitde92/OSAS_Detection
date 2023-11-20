import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import NLLLoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torchinfo import summary
from data_utils import load_data
from scipy.interpolate import splev, splrep
import pickle
from model import ConvNet
from Dataset import OSASUDDataset, ApneaECGDataset

DATA_FILE_NAME = ['normal_segments_sub.pkl', 'disease_segments_sub.pkl']
DATA_DIR = 'data'
MODEL_SAVE_DIR = 'trained_models'
OUTPUT_DIR = 'output'
BATCH_SIZE = 1024
LR = 0.00002
N_EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ir = 3  # interpolate interval
before = 2
after = 2

# normalize
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def plot(train_losses, train_acc, test_losses, test_acc, label):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
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
        y_pred_proba = softmax(y_pred)
        pred = y_pred_proba.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        targets = torch.argmax(target, dim=1, keepdim=True)
        # correct += pred.eq(target.view_as(pred)).sum().item()
        correct += (pred == targets).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f'Loss={running_loss} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
    train_loss = running_loss / len(loader.dataset)
    train_accuracy = 100 * correct / processed
    return train_loss, train_accuracy


def test(model, loader, loss_function):
    model.eval()
    pbar = tqdm(loader)
    running_loss = 0.0
    correct = 0
    processed = 0
    predictions = []
    targets_list = []
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(DEVICE), target.to(DEVICE)
        y_pred = model(data)
        loss = loss_function(y_pred, target)
        running_loss += loss.item()

        # pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        y_pred_proba = softmax(y_pred)
        pred = y_pred_proba.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        targets = torch.argmax(target, dim=1, keepdim=True)
        pred_numpy = pred.detach().cpu().numpy()
        predictions += pred_numpy.tolist()
        target_numpy = targets.detach().cpu().numpy()
        targets_list += target_numpy.tolist()
        # correct += pred.eq(target.view_as(pred)).sum().item()
        correct += (pred == targets).sum().item()
        processed += len(data)
        pbar.set_description(desc=f'Loss={running_loss} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
    test_loss = running_loss / len(loader.dataset)
    test_accuracy = 100 * correct / processed
    conf_matrix = confusion_matrix(targets_list, predictions)
    print(conf_matrix)
    return test_loss, test_accuracy


def load_data_ecg():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(os.path.join('D:\EMBS_Student_Mentoring\Sleep-apnea-detection-through-a-modified-LeNet-5-CNN\dataset', "apnea-ecg.pkl"), 'rb') as f:  # read preprocessing result
        apnea_ecg = pickle.load(f)

    x_train = []
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
        # Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_train.append([rri_interp_signal, ampl_interp_signal])
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1))  # convert to numpy format
    y_train = np.array(y_train, dtype="float32")

    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
        # Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_test.append([rri_interp_signal, ampl_interp_signal])
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    return x_train, y_train, groups_train, x_test, y_test, groups_test


if __name__ == '__main__':
    # classification: 0 = Binary, 1 = Multiclass
    x_train, y_train, x_test, y_test = load_data(DATA_FILE_NAME, classification=0)
    # x_train, y_train, groups_train, x_test, y_test, groups_test = load_data_ecg()
    # ohe = OneHotEncoder(sparse_output=False)
    # y_train_categorical = ohe.fit_transform(y_train.reshape(-1, 1))
    # y_test_categorical = ohe.fit_transform(y_test.reshape(-1, 1))
    # idx = np.random.choice(np.arange(len(x_train)), 100000, replace=False)
    # x_train_sample = x_train[idx]
    # y_train_sample = y_train[idx]
    #
    # idx_test = np.random.choice(np.arange(len(x_test)), 20000, replace=False)
    # x_test_sample = x_test[idx_test]
    # y_test_sample = y_test[idx_test]
    # rfc = RandomForestClassifier(n_estimators=1000, criterion='entropy', n_jobs=6)
    # rfc.fit(x_train.reshape((-1, 5 * 600)), y_train)
    # rfc_preds = rfc.predict(x_test.reshape((-1, 5 * 600)))
    # accuracy = accuracy_score(y_test, rfc_preds)
    # print("Random Forest Accuracy:", accuracy)
    # if np.isnan(x_train).any() or np.isnan(x_test).any():
    #     print("NaN in input")
    #     exit(1)
    # train_dataset = OSASUDDataset(x_train_sample, y_train_sample)
    train_dataset = OSASUDDataset(x_train, y_train)
    # train_dataset = OSASUDDataset(x_train, y_train_categorical)
    # train_dataset = ApneaECGDataset(x_train, y_train_categorical)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_dataset = OSASUDDataset(x_test_sample, y_test_sample)
    test_dataset = OSASUDDataset(x_test, y_test)
    # test_dataset = OSASUDDataset(x_test, y_test_categorical)
    # test_dataset = ApneaECGDataset(x_test, y_test_categorical)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    # num_classes = len(np.unique(y_train))
    num_classes = 2
    print("Number of classes:", num_classes)
    sample_data, sample_target = next(iter(train_loader))
    print(sample_data.size())
    model = ConvNet((sample_data.size()[0], sample_data.size()[1], sample_data.size()[2]), num_classes=num_classes)
    model.to(DEVICE)
    if num_classes > 2:
        # Multi Class classification
        loss_function = CrossEntropyLoss()
        print("Loss Function: Cross Entropy Loss")
    else:
        # Binary Classification
        loss_function = CrossEntropyLoss()
        print("Loss Function: Cross Entropy Loss")

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[70, 120, 170], gamma=0.1)
    # optimizer = SGD(model.parameters(), lr=LR)
    input_size = (sample_data.size()[0], sample_data.size()[1], sample_data.size()[2])
    # summary(model, input_size=input_size)
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
            min_val_loss = test_loss
            print("Validation Loss decreased, saving model")
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(test_acc)
        epoch_valid_loss.append(test_loss)
    plot(epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc,
         f'Loss & Accuracy')
