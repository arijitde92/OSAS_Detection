import pickle
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'
FILE_NAME = 'osasud_numpy_processed.pkl'


def remove_nan(features, labels):
    nan_indices = np.argwhere(np.isnan(features))
    nan_row_numbers = nan_indices[:, 0]
    x_non_nan = np.delete(features, nan_row_numbers, axis=0)
    y_non_nan = np.delete(labels, nan_row_numbers, axis=0)
    return x_non_nan, y_non_nan


def augment(features, labels):
    """
    TODO: SMOTE only takes 2 dimensional data, reshaping only features leads to dimension mismatch with labels. FIX it.
    :param features:
    :param labels:
    :return:
    """
    classes, counts = np.unique(labels, return_counts=True)
    original_feature_shape = (-1,) + features.shape[1:]
    num_samples = features.shape[0]

    print("Before Resampling")
    for idx, cls in enumerate(classes):
        print("Class:", cls, " Count:", counts[idx])
    over_sampler = SMOTE(k_neighbors=10, sampling_strategy='not majority')
    x_aug, y_aug = over_sampler.fit_resample(features.reshape(num_samples, -1), labels)
    print("After Resampling")
    classes, counts = np.unique(y_aug, return_counts=True)
    for idx, cls in enumerate(classes):
        print("Class:", cls, " Count:", counts[idx])
    x_aug = x_aug.reshape(original_feature_shape)
    return x_aug, y_aug


def load_data(file_name, classification=0, test_split=0.33):
    with open(os.path.join(DATA_DIR, file_name), 'rb') as f:  # read preprocessing result
        _, waveform_data, label_data = pickle.load(f)

    x = np.reshape(waveform_data, (-1, 80, 4))
    # x = np.average(x, axis=1)  # Reducing dimension by taking average of the 80 data points
    x = x[:, :, 1:]
    y = np.reshape(label_data, (-1, 2))
    if classification == 0:  # multi class classification
        y = y[:, 1]
    else:  # Choosing binary labels for binary classification
        y = y[:, 0]
    le = LabelEncoder()
    y = le.fit_transform(y)
    x_augmented, y_augmented = augment(*remove_nan(x, y))
    x_train, x_test, y_train, y_test = train_test_split(x_augmented, y_augmented,
                                                        test_size=test_split, random_state=42)
    print("Train data shape:", x_train.shape)
    print("Test data shape:", x_test.shape)
    print("Train label shape:", y_train.shape)
    print("Test label shape:", y_test.shape)
    print("Class wise count:")
    print(np.unique(y, return_counts=True))
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data(FILE_NAME)
