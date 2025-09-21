import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

from exceptions.exceptions import InvalidBackboneError

# num_class = 8  # for SIGNAL-8 dataset
num_class = 24  # for RadioML dataset


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(64, 64, 3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(64, 32, 3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        self.outc = nn.Sequential(
            nn.Conv1d(32, 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.conv1(x1)
        # x3 = self.conv2(x2)
        # x4 = self.conv3(x3)
        out = self.outc(x2)
        return out


# prepare for building the ResNet
class BasicBlock(nn.Module):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(planes)  # the number of output channels
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # in_planes should has the same channels with planes after expansion
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                                                    bias=False),
                                          nn.BatchNorm1d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):  # input size: 2*1024
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        # self.linear = nn.Linear(128 * 500, 128)  # for ResNet(BasicBlock, [2, 2]), SIGNAL-8
        self.linear = nn.Linear(128 * 512, 128)  # for ResNet(BasicBlock, [2, 2]), RadioML

        self.bn2 = nn.BatchNorm1d(128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.bn2(self.linear(out)))  # for SIGNAL-8
        # out = F.relu(self.linear(out))  # for RadioML

        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2])


class CSSL(nn.Module):

    def __init__(self):
        super(CSSL, self).__init__()
        self.backbone = ResNet18()
        self.noise_level_estimator = FCN()

    def forward(self, x):
        noise_level = self.noise_level_estimator(x)
        concat_sig = torch.cat([x, noise_level], dim=1)
        out = self.backbone(concat_sig)

        return out


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = CSSL()  # initialization weight -> init = self.f.state_dict()['backbone.conv1.weight']

        # classifier
        # self.fc = nn.Linear(128, num_class, bias=True)  # for SIGNAL-8
        self.fc = nn.Sequential(nn.Linear(128, 64, bias=True),
                                nn.Linear(64, num_class, bias=True))  # for RadioML

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            self.f.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out
