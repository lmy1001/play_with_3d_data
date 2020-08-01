from __future__ import print_function
import argparse
import os
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from dataset import Mnist3dDataset
from pointnet_cls import PointNetCls, feature_transform_regularizer
from tqdm import tqdm
from pytorchtools import EarlyStopping
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='/Users/lmy/Dataset/mnist_3d/', help='the data directory')
parser.add_argument('--optimizer', type=str, default='Adam',
                    choices=['Adam', 'SGD'], help='choose the optimizer')
parser.add_argument('--momentum', type=float, default=0.9,help='Momentum for SGD')
parser.add_argument('--train_log_name', type=str, default='log/train_log',
                    help='File name of train log event')
parser.add_argument('--val_log_name', type=str, default='log/val_log',
                    help='File name of validation log event')

parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--npoints', type=int, default=2048, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

args = parser.parse_args()

blue = lambda x: '\033[94m' + x + '\033[0m'

args.manualSeed = np.random.randint(1, 10000)  # fix seed
print("Random Seed: ", args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)


train_data = Mnist3dDataset(root=args.data_dir,
                            npoints=args.npoints,
                            split='train')
test_data = Mnist3dDataset(root=args.data_dir,
                           npoints=args.npoints,
                           split='test')
num_train = len(train_data)
print(num_train)

indices = list(range(num_train))
np.random.shuffle(indices)
valid_size = 0.15
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

print('train_sampler: ', len(train_sampler))
print('valid sampler: ', len(valid_sampler))

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    sampler=train_sampler,
    #shuffle=True,
    num_workers=int(args.workers)
)

valid_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    sampler=valid_sampler,
    #shuffle=True,
    num_workers=int(args.workers)
)

test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=int(args.workers),
)

print(len(train_data), len(test_data))
num_classes = 10
num_batch = int(len(train_sampler) / args.batch_size)
val_num_batch = int(len(valid_sampler) / args.batch_size)
print(num_batch, val_num_batch)

try:
    os.makedirs(args.outf)
except OSError:
    pass

cls = PointNetCls(k=num_classes, feature_transform=args.feature_transform)

if args.model !='':
    cls.load_state_dict(torch.load(args.model))

if args.optimizer == 'Adam':
    optimizer = optim.Adam(list(cls.parameters()), lr=args.learn_rate,  betas=(0.9, 0.999))
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(list(cls.parameters()), lr=args.learn_rate, momentum=args.momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
cls = cls.to(device)

# output log to writer
writer_train = SummaryWriter(logdir=args.train_log_name)
writer_val = SummaryWriter(logdir=args.val_log_name)
early_stopping = EarlyStopping(patience=10, verbose=True)

for epoch in range(args.nepoch):
    scheduler.step()

    tot_val_loss = 0
    total_correct = 0
    total_valset, total_val_batch = 0, 0
    for i, data in enumerate(train_dataloader, 0):
        step = i + args.batch_size * epoch
        points, target = data
        points = points.numpy().transpose(0, 2, 1)
        points = torch.from_numpy(points).float()
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        cls = cls.train()
        pred, trans, trans_feat = cls(points)
        loss = F.nll_loss(pred, target)
        writer_train.add_scalar('loss', loss, step)

        if args.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(args.batch_size)))


        if i % 10 == 0:
            j, data = next(enumerate(valid_dataloader, 0))
            points, target = data
            points = points.numpy().transpose(0, 2, 1)
            points = torch.from_numpy(points).float()
            points, target = points.to(device), target.to(device)
            cls = cls.eval()
            pred, _, _ = cls(points)
            val_loss = F.nll_loss(pred, target)
            writer_val.add_scalar('loss', val_loss, step)

            tot_val_loss = tot_val_loss + val_loss
            total_val_batch += 1

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_valset += points.size()[0]
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                epoch, j, val_num_batch, blue('val'), val_loss.item(), correct.item() / float(args.batch_size)))

    tot_val_loss = tot_val_loss / float(total_val_batch)
    early_stopping(tot_val_loss, cls)
    if early_stopping.early_stop:
        print("Early Stopping: %d" % epoch)

        total_correct = 0
        total_testset = 0
        for i, data in tqdm(enumerate(test_dataloader, 0)):
            points, target = data
            points = points.numpy().transpose(0, 2, 1)
            points = torch.from_numpy(points).float()
            points, target = points.to(device), target.to(device)
            cls = cls.eval()
            pred, _, _ = cls(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]
        print("best accuracy {}".format(total_correct / float(total_testset)))
        break

torch.save(cls.state_dict(), '%s/cls_last_model_%d.pth' % (args.outf, epoch))
writer_train.close()
writer_val.close()
