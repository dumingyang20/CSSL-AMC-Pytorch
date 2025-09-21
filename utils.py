import os
import shutil
import numpy as np
import torch
import yaml


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer


def split_data(data, mode=None):

    if mode == 'train':  # 60%
        np.random.seed(20)
        np.random.shuffle(data)
        data_info = data[:int(0.6 * len(data))]

    elif mode == 'test':  # 20% = 60%->80%
        np.random.seed(60)
        np.random.shuffle(data)
        data_info = data[int(0.6 * len(data)):int(0.7 * len(data))]

    else:  # 20% = 80%->100%
        data_info = data[int(0.7 * len(data)):]

    return data_info


def get_real_imag(x, transpose=False):
    real_part = [x[i].real for i in range(len(x))]
    imag_part = [x[i].imag for i in range(len(x))]
    if transpose is False:
        return np.concatenate((real_part, imag_part)).reshape(2, -1)
    else:
        return np.concatenate((real_part, imag_part)).reshape(-1, 2)

