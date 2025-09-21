import torch
import argparse
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from matplotlib import pyplot
import torch.nn as nn

from dataset import Dataset_complex_single, Dataset_IQ_single
from models.cssl import Net

# matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--test_data_dir',
                    required=True,
                    type=str, help='test data')
parser.add_argument('--num_class',
                    # default=8,
                    default=24,  # RadioML dataset
                    type=int, help='the number of category')
parser.add_argument('--encoder_dir',
                    default=None,
                    help='The pretrained encoder path')
parser.add_argument('--linear_model_dir',
                    required=True, # 'linear_model.pth.tar' dir
                    type=str, help='classifier')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--filename',
                    default='GOLD_XYZ_1024_2dB.hdf5',
                    type=str, help='test data')
args = parser.parse_args()

"""load model to test another data set"""
model = Net(num_class=args.num_class, pretrained_path=args.encoder_dir).cuda()
# pre_trained = model.state_dict()['f.backbone.conv1.weight']

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {params}")

# print('SNR=%s' % args.test_data_dir.split('/')[-1])  # for SIGNAL-8
print('SNR=%s' % args.filename.split('.')[0].split('_')[-1])  # for RadioML

print('1. loading model ...')
model_CKPT = torch.load(args.linear_model_dir)
model.load_state_dict(model_CKPT)
model.eval()
print('1. Finish loading !')

"""prepare the test dataset"""
print('2. loading data ...')
# test_signal = Dataset_complex_single(args.test_data_dir, mode='test')  # SIGNAL-8
test_signal = Dataset_IQ_single(args.test_data_dir, filename=args.filename, mode='test')  # RadioML

test_signal_dataloader = DataLoader(test_signal, batch_size=args.bs, shuffle=False)
print('2. Finish loading !')

idx = 0
correct = 0.0
total_test = 0.0
test_acc_list = []
snr_before_list = []
snr_after_list = []
confusion_matrix = torch.zeros(int(args.num_class), int(args.num_class))
for inputs, labels in test_signal_dataloader:
    inputs = inputs.cuda()
    labels = labels.cuda()

    # classify
    output = model(inputs)
    pred = output.detach().max(1)[1]  # classify accuracy
    correct += pred.eq(labels.view_as(pred)).sum()
    total_test += labels.size(0)

    idx += 1

    acc = correct / total_test
    # print('test accuracy in %03d batch：%.2f%%' % (idx, (100 * acc)))
    test_acc_list.append(acc)

    for t, p in zip(labels.view(-1), pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    # plt results
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['pdf.fonttype'] = 42

print('average value of test accuracy：%.2f%%' % (100 * sum(test_acc_list) / len(test_acc_list)))
