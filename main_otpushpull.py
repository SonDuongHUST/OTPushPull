import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from losses import OTPushPullLoss
from network import Encoder, KantorovichNetwork


def parse_option():
    parser = argparse.ArgumentParser('OT Push Pull to learn representations')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--numbers_c', type=int, default=100,
                        help='Numbers of c')
    parser.add_argument('--feature_dim', type=int, default=128,
                        help='Dimensions of image representation')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--encoder', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # other settings
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large training batch')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets'
    opt.model_path = './save/OTPushPull/{}_models'.format(opt.dataset)
    opt.tb_path = './save/OTPushPull/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.path == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform, download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform, download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                             transform=train_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    encoder = Encoder(arch='resnet18')
    kan_net1 = KantorovichNetwork()
    kan_net2 = KantorovichNetwork()
    criterion = OTPushPullLoss(lamda=0.1, alpha=0.9)
    c = torch.rand(opt.numbers_c, opt.feature_dim)
    c = nn.functional.normalize(c, dim=1)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        kan_net1 = kan_net1.cuda()
        kan_net2 = kan_net2.cuda()
        criterion = criterion.cuda()
        c = c.cuda()
        cudnn.benchmark = True

    return encoder, kan_net1, kan_net2, c, criterion


def train(train_loader, encoder, kan_net1, kan_net2, c, criterion, optimizer_min, optimizer_max, epoch, opt):
    """one epoch training"""
    encoder.train()
    kan_net1.train()
    kan_net2.train()
    c.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer_min, optimizer_max)
        # compute loss
        features = encoder(images)
        kantorovich_value_c = kan_net1(c)
        kantorovich_value_z = kan_net2(features)
        loss = criterion(c, features, kantorovich_value_c, kantorovich_value_z)

        # update metric
        losses.update(loss.item(), opt.batch_size)

        # optimization
        # Maximizing loss phase
        for i in range(5):
            optimizer_max.zero_grad()
            loss.backward()
            optimizer_max.step()

        # Minimizing loss phase
        optimizer_min.zero_grad()
        loss.backward()
        optimizer_min.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    encoder, kan_net1, kan_net2, c, criterion = set_model(opt)

    # build optimizer
    optimizer_min = torch.optim.SGD([encoder.parameters(), c],
                                    lr=opt.learning_rate,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
    optimizer_max = torch.optim.SGD([kan_net1.parameters(), kan_net2.parameters()],
                                    lr=opt.learning_rate,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer_min, optimizer_max, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, encoder, kan_net1, kan_net2, c, criterion, optimizer_min, optimizer_max, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer_min.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(encoder, kan_net1, kan_net2, optimizer_min, optimizer_max, opt, epoch, save_file)

        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last.pth')
        save_model(encoder, kan_net1, kan_net2, optimizer_min, optimizer_max, opt, epoch, save_file)


if __name__ == '__main__':
    main()
