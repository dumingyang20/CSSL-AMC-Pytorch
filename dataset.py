import os
import torch
import pickle
from torch.utils.data.dataset import Dataset
import numpy as np
from glob import glob
from utils import get_real_imag, split_data
import h5py
from sklearn import preprocessing


def load_data_complex(root, length):
    """
    :param root: data set direction
    :param length: fix the time-series length
    :return:
    """
    name2label = {}
    labels = []
    raw_data = []
    # 遍历根目录下的子文件夹，并排序，保证映射关系固定（所以每一个类要放到一个文件夹中）
    for name in sorted(os.listdir(os.path.join(root))):
        # 跳过非文件夹
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())

    IF_data = []
    for name in name2label.keys():
        IF_data += glob(os.path.join(root, name, '*.txt'))

    for index in IF_data:
        data = []  # save data
        label = []  # save label
        with open(index, 'rb') as file_pi:
            x = pickle.load(file_pi)  # all signal samples in a single file
            for idx in range(len(x)):
                x[idx] = get_real_imag(x[idx][0:length], transpose=False)
            data.extend(x)
            keys = index.split('/')[-2]
            label.extend((np.ones(len(data))*name2label[keys]).astype(int).tolist())

        raw_data.extend(data)
        labels.extend(label)

    np.random.seed(116)
    np.random.shuffle(raw_data)
    np.random.seed(116)
    np.random.shuffle(labels)

    return raw_data, labels


# load RadioML I/Q data (a pkl file)
def load_RadioML_data(root, filename, type=None):
    if type == 'pkl':
        with open(os.path.join(root, filename), 'rb') as fo:
            list_data = pickle.load(fo, encoding='bytes')

        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], list_data.keys())))),
                         [1, 0])  # different SNRs and mods
        data = []
        lbl = []
        for mod in mods:
            for snr in snrs:
                data.append(list_data[(mod, snr)])
                for i in range(list_data[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
        data = np.vstack(data)  # stack data with specific SNR
        label = encoder(lbl, mods)  # encode labels (not one-hot)

    elif type == 'hdf5':
        f = h5py.File(os.path.join(root, filename), 'r')
        data = list(f['X'])
        label = list(f['Y'])
        # normalization
        # data = normalization(data)

    return data, label


def encoder(labels, classes):
    yy1 = []
    for i in range(len(labels)):
        yy1.append(classes.index(labels[i][0]))

    return yy1


def get_index(lst=None, item=None):
    return [index for (index, value) in enumerate(lst) if value == item]


def load_specific_label(root, filename, idx):
    f = h5py.File(os.path.join(root, filename), 'r')
    data = list(f['X'])
    labels = list(f['Y'])

    labels_all = [int(i) for i in labels]
    indices = get_index(labels_all, idx)

    data = [data[i] for i in indices]
    label = [labels_all[i] for i in indices]

    return data, label

# def normalization(data):
#     min_max_scaler = preprocessing.MinMaxScaler()
#     for i in range(len(data)):
#         data[i] = min_max_scaler.fit_transform(data[i])
#
#     return data


class Dataset_complex(Dataset):
    def __init__(self, root_dir, mode=None):
        """
        complex radar signals
        :param root_dir: data set direction
        :param mode: 'train' or 'test'
        """
        self.data_info, self.label_info = [], []
        for dir in root_dir:
            data_ = load_data_complex(dir, length=1000)
            data, label = split_data(data_[0], mode=mode), split_data(data_[1], mode=mode)
            data, label = torch.Tensor(data), torch.LongTensor(label)
            self.data_info.append(data)
            self.label_info.append(label)

    def __getitem__(self, idx):
        img, target = [x[idx] for x in self.data_info], [y[idx] for y in self.label_info]

        return img, target

    def __len__(self):
        return len(self.label_info[0])


class Dataset_IQ(Dataset):
    def __init__(self, root_dir, filename, mode=None):
        """
        I/Q radar signals: RadioML data
        :param root_dir: data set direction
        :param rate: the proportion of train sample, default >= 0.6
        """
        self.data_info, self.label_info = [], []
        for file in filename:
            data_ = load_RadioML_data(root_dir, filename=file, type='hdf5')
            data, label = split_data(data_[0], mode=mode), split_data(data_[1], mode=mode)
            data, label = torch.Tensor(data), torch.LongTensor(label)
            self.data_info.append(data)
            self.label_info.append(label)

    def __getitem__(self, idx):
        img, target = [x[idx] for x in self.data_info], [y[idx] for y in self.label_info]
        return img, target

    def __len__(self):
        return len(self.label_info[0])


class Dataset_complex_single(Dataset):
    def __init__(self, root_dir, mode=None):
        """
        complex radar signals
        :param root_dir: data set direction
        :param mode: 'train' or 'test'
        """
        self.root = root_dir
        data_ = load_data_complex(self.root, length=1000)
        data, label = split_data(data_[0], mode=mode), split_data(data_[1], mode=mode)
        self.data_info, self.label_info = torch.Tensor(data), torch.LongTensor(label)

    def __getitem__(self, idx):
        img, target = self.data_info[idx], self.label_info[idx]

        return img, target

    def __len__(self):
        return len(self.label_info)


class Dataset_IQ_single(Dataset):
    def __init__(self, root_dir, filename, mode=None):
        """
        I/Q radar signals: RadioML data
        :param root_dir: data set direction
        :param rate: the proportion of train sample, default >= 0.6
        """
        self.root = root_dir
        self.data, self.label = load_RadioML_data(self.root, filename=filename, type='hdf5')
        self.data_info, self.label_info = split_data(self.data, mode=mode), split_data(self.label, mode=mode)
        self.data_info, self.label_info = torch.Tensor(np.array(self.data_info)), torch.Tensor(self.label_info)

    def __getitem__(self, idx):
        return self.data_info[idx], self.label_info[idx]

    def __len__(self):
        return len(self.label_info)

