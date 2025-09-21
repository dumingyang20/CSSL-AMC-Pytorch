import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from torch.utils.data import DataLoader
import os

from dataset import Dataset_IQ, Dataset_complex
from models.cssl import CSSL
from contrastive_learning import Contrastive_learning

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser(description='CSSL-AMC')
parser.add_argument('-data', metavar='DIR',
                    required=True,
                    help='path to dataset')
parser.add_argument('-j', '--workers',
                    default=0,
                    type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs',
                    default=200,
                    type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size',
                    default=256,
                    type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate',
                    default=0.0003,
                    type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    default=False,
                    help='Whether or not to use 16-bit precision GPU training.'
                    )
parser.add_argument('--out_dim',
                    default=128,
                    type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps',
                    default=10,
                    type=int,
                    help='Log every n steps')
parser.add_argument('--temperature',
                    default=0.07,
                    type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views',
                    default=2,
                    type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--log_dir', default='./runs/finetune/1',
                    type=str, help='logging directory.')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # signal dataset
    train_dataset = Dataset_IQ(args.data, ['GOLD_XYZ_1024_30dB.hdf5',
                                           'GOLD_XYZ_1024_4dB.hdf5',
                                           ], mode='train')  # 1

    # train_dataset = Dataset_IQ(args.data, ['GOLD_XYZ_1024_30dB.hdf5',
    #                                        'GOLD_XYZ_1024_0dB.hdf5',
    #                                        ], mode='train')  # 2

    # train_dataset = Dataset_IQ(args.data, ['GOLD_XYZ_1024_0dB.hdf5',
    #                                        'GOLD_XYZ_1024_-6dB.hdf5',
    #                                        ], mode='train')  # 3

    # train_loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    model_q = CSSL()
    model_k = CSSL()

    params_q = sum(p.numel() for p in model_q.parameters() if p.requires_grad)
    params_k = sum(p.numel() for p in model_k.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {params_q + params_k}")

    optimizer = torch.optim.Adam(model_q.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    momentum = 0.999
    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        Con_learn = Contrastive_learning(model_q=model_q, model_k=model_k, optimizer=optimizer,
                        scheduler=scheduler, momentum=momentum, args=args)
        Con_learn.train(train_loader)


if __name__ == "__main__":
    main()
