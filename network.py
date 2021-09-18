import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet


class KantorovichNetwork(nn.Module):
    def __init__(self, embeddings_size=128, output_size=1):
        super(KantorovichNetwork, self).__init__()
        self.fc1 = nn.Linear(embeddings_size, embeddings_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(embeddings_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Encoder(nn.Module):
    """
    Common CIFAR ResNet recipe
    Comparing to ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, stride=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None):
        super(Encoder, self).__init__()
        self.net = []
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # not normalized here
        return x
