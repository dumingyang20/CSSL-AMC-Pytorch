from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets
from tqdm import tqdm
import torch
import argparse
import torch.nn.functional as F
import pandas as pd
import os

from utils import adjust_learning_rate, accuracy
from dataset import Dataset_IQ, Dataset_complex
from models.cssl import CSSL, Net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Fine-tune SimCLR')

parser.add_argument('-data', metavar='DIR',
                    required=True,  # RadioML dataset
                    help='path to dataset')
parser.add_argument('--epochs',
                    default=200,
                    type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--bs',
                    default=256,
                    type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate',
                    default=0.0003,
                    type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--num_class',
                    # default=8,  # SIGNAL-8
                    default=24,  # RadioML
                    type=int, help='the number of category')
parser.add_argument('--model_path', type=str,
                    default='./runs/finetune/1/checkpoint.pth.tar',
                    help='The pretrained model path')
parser.add_argument('--lr_rate', default=20, type=int, help='lr_update_freq')

args = parser.parse_args()

# load data
train_dataset = Dataset_IQ(args.data, ['GOLD_XYZ_1024_30dB.hdf5',
                                       'GOLD_XYZ_1024_4dB.hdf5',
                                       ], mode='train')  # 1

# train_dataset = Dataset_IQ(args.data, ['GOLD_XYZ_1024_30dB.hdf5',
#                                        'GOLD_XYZ_1024_0dB.hdf5',
#                                        ], mode='train')  # 2

# train_dataset = Dataset_IQ(args.data, ['GOLD_XYZ_1024_0dB.hdf5',
#                                        'GOLD_XYZ_1024_-6dB.hdf5',
#                                        ], mode='train')  # 3

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)


test_dataset = Dataset_IQ(args.data, ['GOLD_XYZ_1024_30dB.hdf5',
                                      'GOLD_XYZ_1024_4dB.hdf5',
                                      ], mode='test')  # 1

# test_dataset = Dataset_IQ(args.data, ['GOLD_XYZ_1024_30dB.hdf5',
#                                       'GOLD_XYZ_1024_0dB.hdf5',
#                                       ], mode='test')  # 2

# test_dataset = Dataset_IQ(args.data, ['GOLD_XYZ_1024_0dB.hdf5',
#                                       'GOLD_XYZ_1024_-6dB.hdf5',
#                                       ], mode='test')  # 3

test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

# load model
model = Net(num_class=args.num_class, pretrained_path=args.model_path).cuda()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().cuda()

results = {'train_acc@1': [],
           'test_acc@1': [],
           'test_acc@5': []}

best_acc = 0.0
top1_train_accuracy = 0
for epoch_counter in range(1, args.epochs + 1):
    optimizer = adjust_learning_rate(optimizer, epoch_counter, args.lr_rate)
    for counter, (images, labels) in enumerate(train_loader):
        model.train()
        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        images, labels = images.cuda(), labels.cuda()

        logits = model(images)
        loss = criterion(logits, labels)
        top1 = accuracy(logits, labels, topk=(1,))
        top1_train_accuracy += top1[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test weight update

    top1_train_accuracy /= (counter + 1)
    results['train_acc@1'].append(top1_train_accuracy.item())

    top1_accuracy = 0
    top5_accuracy = 0
    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            model.eval()
            x_batch = torch.cat(x_batch, dim=0)
            y_batch = torch.cat(y_batch, dim=0)
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

            logits = model(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

        results['test_acc@1'].append(top1_accuracy.item())
        results['test_acc@5'].append(top5_accuracy.item())

        print(
            f"Epoch {epoch_counter}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch_counter + 1))
        data_frame.to_csv('linear_statistics.csv', index_label='epoch')

        if top1_accuracy.item() > best_acc:
            best_acc = top1_accuracy.item()
            torch.save(model.state_dict(), 'linear_model.pth.tar')
