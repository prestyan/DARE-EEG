import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from scipy.signal import resample
import os

class EegDataset(Dataset):
    def __init__(self, data, labels, source_or_target='s', transform=None):
        self.transform = transform
        if source_or_target == 's':
            self.eeg_data = torch.from_numpy(data).float()
            # self.eeg_data = self.eeg_data.permute(2,0,1)
            self.labels = torch.from_numpy(labels).long()
        elif source_or_target == 't':
            self.eeg_data = torch.from_numpy(data).float()
            # self.eeg_data = self.eeg_data.permute(2,0,1)
            self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.eeg_data[index]
        if self.transform is not None:
            x = self.transform(x)
        y = self.labels[index]
        return x, y

class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y

class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y

def prepare_TUAB_dataset(root):
    # set random seed
    seed = 2025
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def prepare_TUEV_dataset(root):
    # set random seed
    seed = 2025
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_test"), test_files
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset
    
def generate_dataload(mode='single', 
                      data_path_1='E:/data/EEG/Sub_S1_single/Sub_S1_', 
                      data_path_2='E:/data/EEG/Sub_S1_single/Sub_S1_',
                      index=1, batch_size=64):
    # input data should be (B, C, T)
    assert mode in ['single', 'mix']

    if mode == 'single':
        # subject-dependent strategy
        data = scipy.io.loadmat(data_path_1 + str(index) + '.mat')
        S1Data = data['S1Data']  # shape: (61, 500, 447)
        S1Label = data['S1Label']  # shape: (447, 1)
        S1Data = np.transpose(S1Data, (2, 0, 1))

        X_train, X_test, y_train, y_test = train_test_split(S1Data, S1Label, test_size=0.2, random_state=2025)

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

        train_dataset = EegDataset(X_train, y_train)
        test_dataset = EegDataset(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, test_loader
    else:
        # cross-subject strategy

        # generate source domain
        data_source = scipy.io.loadmat(data_path_1 + str(index) + '.mat')
        X_train = data_source['S1Data']  # shape: (61, 500, 447)
        y_train = data_source['S1Label']  # shape: (447, 1)
        X_train = np.transpose(X_train, (2, 0, 1))
        y_train = np.squeeze(y_train)

        train_dataset = EegDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # generate target domain
        data_target = scipy.io.loadmat(data_path_2 + str(index) + '.mat')
        X_test = data_target['S1Data']
        y_test = data_target['S1Label']
        X_test = np.transpose(X_test, (2, 0, 1))
        y_test = np.squeeze(y_test)

        test_dataset = EegDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, test_loader

def generate_dataset_tuab(dataset='TUAB',
                           data_path='F:/data/TUAB/tuh_eeg_abnormal/edf/processed',
                           index=1, batch_size=64):
    assert dataset in ['TUAB', 'TUEV']
    train_dataset, test_dataset, val_dataset = None, None, None
    ch_names, metrics = None, None
    if dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = prepare_TUAB_dataset(data_path)
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        # nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    elif dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = prepare_TUEV_dataset(data_path)
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        # nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    '''
    elif dataset == 'TUSZ':
        train_dataset, test_dataset, val_dataset = prepare_TUSZ_dataset(None)

        # ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
        #             'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        # ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                    'T4', 'T6', 'O2']
        # nb_classes = 8
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    '''
    return train_dataset, test_dataset, val_dataset, ch_names, metrics







