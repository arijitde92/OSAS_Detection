import pickle
from typing import Dict

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
import numpy as np
import os

from numpy import ndarray
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'
FILE_NAMES = ['normal_segments_sub.pkl', 'disease_segments_sub.pkl']


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


def load_data(file_names, classification=0, test_split=0.33):
    with open(os.path.join(DATA_DIR, file_names[0]), 'rb') as f:  # read preprocessing result
        if 'normal' in file_names[0]:
            normal_data_dict = pickle.load(f)
        else:
            disease_data_dict = pickle.load(f)
    with open(os.path.join(DATA_DIR, file_names[1]), 'rb') as f:  # read preprocessing result
        if 'normal' in file_names[1]:
            normal_data_dict = pickle.load(f)
        else:
            disease_data_dict = pickle.load(f)
    # normal_derived_data = normal_data_dict['derived']
    # normal_waveform_data = normal_data_dict['waveform']
    normal_psg_data = normal_data_dict['psg']
    normal_labels = normal_data_dict['label']

    # disease_derived_data = disease_data_dict['derived']
    # disease_waveform_data = disease_data_dict['waveform']
    disease_psg_data = disease_data_dict['psg']
    disease_binary_labels = disease_data_dict['binary_label']
    disease_multi_labels = disease_data_dict['multi_label']

    # x = np.reshape(waveform_data, (-1, 80, 4))
    # x = np.average(x, axis=1)  # Reducing dimension by taking average of the 80 data points

    # Combine disease and normal data and then shuffle them
    # derived_data = np.concatenate((normal_derived_data, disease_derived_data), axis=0)
    # waveform_data = np.concatenate((normal_waveform_data, disease_waveform_data), axis=0)
    psg_data = np.concatenate((normal_psg_data, disease_psg_data), axis=0)
    binary_labels = np.concatenate((normal_labels, disease_binary_labels), axis=0)
    multi_labels = np.concatenate((normal_labels, disease_multi_labels), axis=0)

    # Shuffle data
    p = np.random.permutation(len(psg_data))
    # derived_data = derived_data[p]
    # waveform_data = waveform_data[p]
    psg_data = psg_data[p]
    binary_labels = binary_labels[p]
    multi_labels = multi_labels[p]

    # Choosing PSG data
    X = psg_data
    if classification == 1:
        # le = LabelEncoder()
        # y = le.fit_transform(multi_labels)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        y = ohe.fit_transform(np.reshape(multi_labels, (-1, 1)))
    else:  # Choosing binary labels for binary classification
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        y = ohe.fit_transform(np.reshape(binary_labels, (-1, 1)))
    # le = LabelEncoder()
    # y = le.fit_transform(y)
    # x_augmented, y_augmented = augment(*remove_nan(x, y))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, shuffle=True)
    print("Train data shape:", x_train.shape)
    print("Test data shape:", x_test.shape)
    print("Train label shape:", y_train.shape)
    print("Test label shape:", y_test.shape)
    print("Class wise count:")
    print(np.unique(y, return_counts=True))
    return x_train, y_train, x_test, y_test


def create_patient_map_features(file_path: str) -> dict:
    # The following code re-assembles the time series related to each patient
    # Loading the dataset
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    # given a patient, the map returns a map that, given feature, returns its whole time series
    patient_map_features = {}
    pbar = tqdm(desc="Processed patients", total=len(dataset['patient'].unique()))
    for pat in dataset['patient'].unique():
        temp = dataset[dataset['patient'] == pat]
        feature_map_ts = {}
        for col in dataset.columns[1:]:
            if 'signal' not in col and 'PSG_' not in col:
                feature_map_ts[col] = temp[col].values
            else:
                feature_map_ts[col] = np.concatenate(temp[col].values)
        patient_map_features[pat] = feature_map_ts
        pbar.update(1)
    pbar.close()
    return patient_map_features


def convert_to_numpy_dataset(map: dict, window_seconds=60) -> dict[str, ndarray]:
    # The following code generates a set of numpy arrays that can be used for machine learning purposes
    # Each row of every numpy array contains information related to windows of length "window_seconds"
    # Here, "PSG_" signal data is not considered

    # window length, in seconds, of each data window.
    # Here, windows are disjoint, but it might be useful to generate overlapping windows so to consider

    derived_data = []  # will be a numpy array containing ECG and PPG derived data
    waveform_data = []  # will be a numpy array containing ECG and PPG waveform data
    psg_data = []  # will be a numpy array containing PSG data
    label_data = []  # will be a numpy array containing the "event" and anomaly" data labels

    list_patients = []  # list that keeps track of the patient ID for each row in the arrays
    list_derived_columns = []  # list that keeps track of the column names in the derived_data numpy array
    list_waveform_columns = []  # list that keeps track of the column names in the waveform_data numpy array
    list_psg_columns = []  # List that keeps track of the column names in the psg_data numpy array (lasy dimension)
    list_label_columns = []  # list that keeps track of the column names in the label_data numpy array (last dimension)

    pbar = tqdm(desc="Processed patients", total=len(map.keys()))
    for pat in map.keys():
        num_values = len(map[pat]['HR(bpm)'])
        max_values = (num_values // window_seconds) * window_seconds
        print("Patient", pat, " >  Discarding the last", num_values - max_values, 'seconds.')

        # Derived data
        temp_list_derived = []
        derived_colnames = []
        for col in map[pat]:
            if np.any([x == col for x in ['HR(bpm)', 'SpO2(%)', 'PI(%)', 'RR(rpm)', 'PVCs(/min)']]):
                temp_list_derived.append(
                    np.asarray(map[pat][col][:max_values]).reshape(-1, window_seconds))
                derived_colnames.append(col)
        temp_list_derived = np.moveaxis(np.asarray(temp_list_derived), [0, 1, 2], [2, 0, 1])
        derived_data.append(temp_list_derived)
        if len(list_derived_columns) == 0:
            list_derived_columns = derived_colnames

        # Waveform data
        temp_list_waveform = []
        waveform_colnames = []
        for col in map[pat]:
            if 'signal' in col:
                temp_list_waveform.append(
                    np.asarray(map[pat][col][:max_values * 80]).reshape(-1, window_seconds, 80))
                waveform_colnames.append(col)
        temp_list_waveform = np.moveaxis(np.asarray(temp_list_waveform), [0, 1, 2, 3], [3, 0, 1, 2])
        waveform_data.append(temp_list_waveform)
        if len(list_waveform_columns) == 0:
            list_waveform_columns = waveform_colnames

        # PSG data
        temp_list_psg = []
        psg_colnames = []
        for col in map[pat]:
            if 'PSG_' in col:
                temp_list_psg.append(
                    np.asarray(map[pat][col][:max_values * 10]).reshape(-1, window_seconds, 10))
                psg_colnames.append(col)
        temp_list_psg = np.moveaxis(np.asarray(temp_list_psg), [0, 1, 2, 3], [3, 0, 1, 2])
        psg_data.append(temp_list_psg)
        if len(list_psg_columns) == 0:
            list_psg_columns = psg_colnames

        # Label data
        temp_list_label = []
        label_colnames = []
        for col in map[pat]:
            if col == 'anomaly' or col == 'event':
                temp_list_label.append(
                    np.asarray(map[pat][col][:max_values]).reshape(-1, window_seconds))
                label_colnames.append(col)
        temp_list_label = np.moveaxis(np.asarray(temp_list_label), [0, 1, 2], [2, 0, 1])
        label_data.append(temp_list_label)
        if len(list_label_columns) == 0:
            list_label_columns = label_colnames

        # Auxiliary data that keeps track of the patient related to each row
        list_patients.extend([pat] * temp_list_derived.shape[0])

        pbar.update(1)
    pbar.close()

    derived_data = np.vstack(derived_data)  # 16008 windows, of 60 seconds each, for 5 attributes
    waveform_data = np.vstack(
        waveform_data)  # 16008 windows, of 60 seconds each, 80 values per second, for 4 attributes
    psg_data = np.vstack(psg_data)  # 16008 windows, of 60 seconds each, 10 values per second, for 5 attributes
    label_data = np.vstack(label_data)  # 16008 windows, of 60 seconds each, for 2 labels

    print(derived_data.shape, waveform_data.shape, psg_data.shape, label_data.shape)
    print(len(list_patients), len(list_derived_columns), len(list_waveform_columns), len(list_psg_columns),
          len(list_label_columns))
    return {'derived_data': derived_data, 'waveform_data': waveform_data, 'psg_data': psg_data,
            'label_data': label_data}


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data(FILE_NAMES)
    # create_patient_map_features('data/dataset_OSAS.pickle')
