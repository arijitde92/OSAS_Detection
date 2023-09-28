DATA_DIR = 'data'

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_name, classification=0, test_split=0.33):
    with open(os.path.join(DATA_DIR, file_name), 'rb') as f:  # read preprocessing result
        _, waveform_data, label_data = pickle.load(f)

    x = np.reshape(waveform_data, (-1, 80, 4))
    y = np.reshape(label_data, (-1, 2))
    if classification == 0:
        y = y[:, 1]  # Choosing binary labels for binary classification, change to 0 for multi class classification
    else:
        y = y[:, 0]
    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=42)
    print("Train data shape:", x_train.shape)
    print("Test data shape:", x_test.shape)
    print("Train label shape:", y_train.shape)
    print("Test label shape:", y_test.shape)
    print("Class wise count:")
    print(np.unique(y, return_counts=True))
    return x_train, y_train, x_test, y_test
