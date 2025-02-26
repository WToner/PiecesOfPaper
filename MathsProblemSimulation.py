import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, utils
from torch.distributions.beta import Beta
import torchvision
import numpy as np
import math
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats
from torch.utils.data import Dataset, DataLoader
from Losses import L2, L1, LabelSmoothing, FullEntropyLoss, BootstrappingLoss, BaselineLoss, GreedyCE, ProbSpherical, RenyiEntropy, NoiseLoss, TruncatedLoss, EntropyLoss, CorrectionBootstrap, CorrectionLoss, ExperimentalLoss, ExperimentalLoss2, ExperimentalLoss3, ExperimentalLoss4,  ExperimentalLoss5, FlippedCrossEntropy, MixUpCE, JensenShannonDivergenceWeightedScaled, CrossEntropy, NCE_RCE, NCE_MAE, AdjCal, Lp, LOOL_CE, Sigmoid,  SCE, GCE,LOOL_L1, MultiMarginLoss, PosDefLoss, GenericLoss, BregmanParamatersQuadratic, UniformVarianceLoss, AdvCal, AdvJensenShannon, JensenShannon, L2_squared, MultiUniformVarianceLoss, Spherical, HS
from LossDataset import Flights, Periodic, NoisedFashion, NoisedEMNIST, BatchSampler, OrderedCIFAR10, CroppedMNIST, Simple, NoisedTinyImageNet, Animals, NoisedMiniWebvision, TwoDimDataset, NoisedCroppedMNIST, NoisedMNIST, NoisedCIFAR10, NoisedCIFAR100, NoisedToy, Shapes, NoisyShapes
import os
from LossDataset import get_cifar, get_cifar10_transforms

"""Here we look at a problem which I can't solve and so I've opted to simulate. We imagine that we draw two elements from 
a distribution and take a mean. Next we draw another (secret) point from the same distribution. We are told whether it is above 
or below this mean. Our goal is that in general the remaining probability mass containing the secret point is minimised."""

latent_dim = 10

class ToyClassifier(nn.Module):
    def __init__(self, input_dim = 1, output_dim=1):
        super(ToyClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, 500)
        self.fc2 = nn.Linear(500, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, output_dim)

        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(200)


    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)

        x = self.fc1(x.float())
        x = F.leaky_relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.bn3(x)

        x = self.fc4(x)

        if x.size()[1] == 1:
            return F.sigmoid(x), x
        else:
            out = F.softmax(x, dim=1), x
            return out

"""A generator with a convolutional structure"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 1)
        #self.bn3 = nn.BatchNorm2d(10)
        #self.conv4 = nn.Linea()

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, latent_dim)

        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        """x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.tanh(x)"""

        #x = x.view(batch_size, 784)
        return x

def train(args, generator, device, optimizer, epoch=0):
    """Train classifier on dataset using loss_function, record output of other loss functions.
    """
    generator.train()
    total_loss = 0
    for i in range(1000):
        noise1 = torch.randn(args.test_batch_size, latent_dim)
        noise2 = torch.randn(1, latent_dim)
        noise3 = torch.randn(1, latent_dim)

        #the secret element is the first of generated1
        prior = torch.ones(args.test_batch_size)/args.test_batch_size
        generated1 = generator(noise1)
        generated2 = generator(noise2)
        generated3 = generator(noise3)
        secret = generated1[0,:].unsqueeze(dim=0).detach()

        distance_s2 = torch.norm(secret - generated2)
        distance_s3 = torch.norm(secret - generated3)
        if distance_s2 < distance_s3:
            distances12 = torch.norm(generated1 - generated2 ,dim=1)
            distances13 = torch.norm(generated1 - generated3 ,dim=1)
            ratio = distances13/(distances12+distances13)
        else:
            distances12 = torch.norm(generated1 - generated2 ,dim=1)
            distances13 = torch.norm(generated1 - generated3 ,dim=1)
            ratio = distances12/(distances12+distances13)
        prior = prior * ratio
        prior = prior/(prior.sum(dim=0))

        loss = -prior[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        #i += 1
    print(total_loss)

def test(args, generator, device, epoch=0):
    """Evaluate classifier on un-noised data and output accuracy and various losses.
    """
    generator.eval()
    total_loss = 0
    correct = 0
    i = 0
    with torch.no_grad():
        for i in range(100):
            noise1 = torch.randn(args.test_batch_size, latent_dim)
            noise2 = torch.randn(1, latent_dim)
            noise3 = torch.randn(1, latent_dim)

            #the secret element is the first of generated1
            prior = torch.ones(args.test_batch_size)/args.test_batch_size
            secret = generator(noise1[0,:].unsqueeze(dim=0))
            generated1 = generator(noise1)
            generated2 = generator(noise2)
            generated3 = generator(noise3)

            distance_s2 = torch.norm(secret - generated2)
            distance_s3 = torch.norm(secret - generated3)
            if distance_s2 < distance_s3:
                distances12 = torch.norm(generated1 - generated2 ,dim=1)
                distances13 = torch.norm(generated1 - generated3 ,dim=1)
                ratio = distances13/(distances12+distances13)
            else:
                distances12 = torch.norm(generated1 - generated2 ,dim=1)
                distances13 = torch.norm(generated1 - generated3 ,dim=1)
                ratio = distances12/(distances12+distances13)
            prior = prior * ratio
            prior = prior/(prior.sum(dim=0))

            loss = -prior[0]
            if i == 0:
                plt.hist(generated1.detach().numpy())
                plt.show()
                stats.probplot(generated1.squeeze(dim=1).detach().numpy(), dist="norm", plot=pylab)
                pylab.show()
            i += 1


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=300, metavar='N',
    help='input batch size for training (default: 64)')
    parser.add_argument('--weight-decay', type=float, default=0.01, metavar='N',
    help='input weight decay (default: 0.01)')
    parser.add_argument('--test-batch-size', type=int, default=400, metavar='N',
    help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=51, metavar='N',
    help='number of epochs to train (default: 30)')
    parser.add_argument('--loss', type=int, default=1, metavar='N',
                        help='code for the loss type to use')
    parser.add_argument('--loss2', type=int, default=-1, metavar='N',
                        help='code for the second loss type to use if any')
    parser.add_argument('--loss_name', choices=['ce', 'l2', 'logit_sph', 'prob_sph', 'l1', 'l0.9', 'sce', 'nce_mae', 'nce_rce', 'label_smooth', 'mixup_ce'], default='ce')
    parser.add_argument('--dataset', choices=['NoisedMNIST', 'NoisedCroppedMNIST', 'Periodic', 'NoisedEMNIST', 'NoisedEMNISTASYM', 'NoisedFashion', 'Animals', 'NoisedCIFAR10', 'NoisedToy', 'NoisyShapes', 'NoisedCIFAR100', 'OrderedCIFAR10', 'NoisedCIFAR100ASYM', 'WebVision', 'TinyImageNet'], default='Periodic')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
    help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    help='SGD momentum (default: 0.5)')
    parser.add_argument('--renyi', type=float, default=0.5, metavar='M',
    help='Alpha for use in renyi loss (default: 0.5)')
    parser.add_argument('--label_noise_rate', type=float, default=0.2, metavar='M',
    help='Proportion of Labels Randomly Flipped (default: 0.1)')
    parser.add_argument('--period', type=float, default=1, metavar='M',
    help='Period of the periodic toy dataset (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
    help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
    help='random seed (default: 1)')
    parser.add_argument('--save', type=int, default=0, metavar='S',
    help='save the model parameters after training? (default: 0)')
    parser.add_argument('--mixup', type=int, default=0, metavar='S',
    help='whether to use mix-up augmentation? (default: 0)')
    parser.add_argument('--load', type=int, default=0, metavar='S',
    help='load the model parameters before training? (default: 0)')
    parser.add_argument('--cuda_no', type=int, default=0, metavar='S',
    help='gpu to train on (default: 6)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
    help='For Saving the current Model')
    #Below are arguments just for the GJS loss
    parser.add_argument('--augs', help='string of n(no), w(weak) and s(strong) augs', type=str, default='s')
    parser.add_argument('--augmentation', type=str, help='which type of strong augmentation to use', choices=['rand'],
                        default='rand')
    parser.add_argument('--N', type=int, help='HP for RandAugment', default=1)
    parser.add_argument('--M', type=int, help='HP for RandAugment', default=3)
    parser.add_argument('--cutout', type=int, help='length used for cutout, 0 disables it', default=16)
    parser.add_argument('--asym', help='Assymetric noise', action='store_false', default=False)
    parser.add_argument('--js_weights', help='First weight is for label, the next are in the order of "augs"',
                           type=str, default='0.5 0.5')
    parser.add_argument('-a', '--arch', type=str,
                           choices=['resnet18', 'resnet34', 'resnet50'],
                           default='resnet18')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #use_cuda = True #
    device = torch.device("cuda:"+str(args.cuda_no) if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    generator = Generator().to(device)
    optimizer = optim.SGD(generator.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120], gamma=0.6)

    for epoch in range(1, args.epochs):
        train(args, generator,  device, optimizer, epoch=epoch)
        scheduler.step()
        if epoch % 1 == 0:
            test(args, generator, device, epoch=epoch)

if __name__ == '__main__':
    main()
