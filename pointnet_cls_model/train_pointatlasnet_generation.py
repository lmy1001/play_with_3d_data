from __future__ import print_function
import argparse
import os
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch3d.loss import chamfer_distance
import numpy as np
from dataset import Mnist3dDataset
from tqdm import tqdm
from pytorchtools import EarlyStopping
from tensorboardX import SummaryWriter
from PointAtlasnet import PointAtlasnet
import visdom


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='/Users/lmy/Dataset/mnist_3d/', help='the data directory')
parser.add_argument('--optimizer', type=str, default='Adam',
                    choices=['Adam', 'SGD'], help='choose the optimizer')
parser.add_argument('--momentum', type=float, default=0.9,help='Momentum for SGD')
parser.add_argument('--train_log_name', type=str, default='pointatlasnet/log/train_log',
                    help='File name of train log event')
parser.add_argument('--val_log_name', type=str, default='pointatlasnet/log/val_log',
                    help='File name of validation log event')

parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--npoints', type=int, default=2048, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='pointatlasnet/cls', help='output folder')
parser.add_argument('--model', type=str, default='pointatlasnet/cls/cls_last_model_39.pth', help='model path')

parser.add_argument('--input_dim', type=int, default=3, help='input/output data dimension')
parser.add_argument('--num_layers', type=int, default=2, help='number of hidden MLP Layer')
parser.add_argument('--hidden_neurons', type=int, default=512, help='number of neurons in each hidden layer')
parser.add_argument('--bottleneck_size', type=int, default=1024, help='dim_out_patch')
parser.add_argument('--activation', type=str, default='relu',
                    choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"], help='dim_out_patch')

args = parser.parse_args()

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
test_num_batch = int(len(test_data) / args.batch_size)
print(num_batch, val_num_batch, test_num_batch)

try:
    os.makedirs(args.outf)
except OSError:
    pass

net = PointAtlasnet(input_dim=args.input_dim, output_dim=args.input_dim,
                    bottleneck_size=args.bottleneck_size, num_layers=args.num_layers,
                    hidden_neurons=args.hidden_neurons, activation=args.activation)



if args.optimizer == 'Adam':
    optimizer = optim.Adam(list(net.parameters()), lr=args.learn_rate,  betas=(0.9, 0.999))
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(list(net.parameters()), lr=args.learn_rate, momentum=args.momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

if args.model !='':
    ckpt = torch.load(args.model)
    net.load_state_dict(ckpt['model'], strict=True)
    start_epoch = ckpt['epoch']
else:
    start_epoch = 0

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

# output log to writer
writer_train = SummaryWriter(logdir=args.train_log_name)
writer_val = SummaryWriter(logdir=args.val_log_name)
early_stopping = EarlyStopping(patience=8, verbose=True, path='pointatlasnet/checkpoint.pt')

for epoch in range(start_epoch, args.nepoch):
    scheduler.step()

    total_val_loss = 0
    total_test_loss = 0
    total_val_batch = 0
    net.train()
    for i, data in enumerate(train_dataloader, 0):
        step = i + args.batch_size * epoch
        points, _ = data
        points = points.numpy().transpose(0, 2, 1)
        points = torch.from_numpy(points).float()
        target = points
        optimizer.zero_grad()
        gen = net(points)

        gen = torch.transpose(gen, 1, 2)
        target = torch.transpose(target, 1, 2)
        chamfer_loss, _ = chamfer_distance(gen, target)

        writer_train.add_scalar('loss', chamfer_loss, step)

        chamfer_loss.backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, chamfer_loss.item()))

        if i % 10 == 0:
            j, data = next(enumerate(valid_dataloader, 0))
            points, _ = data
            points = points.numpy().transpose(0, 2, 1)
            points = torch.from_numpy(points).float()
            target = points
            net.eval()
            with torch.no_grad():
                gen = net(points)

            gen = torch.transpose(gen, 1, 2)
            target = torch.transpose(target, 1, 2)
            val_chamfer_loss, _ = chamfer_distance(gen, target)
            writer_val.add_scalar('loss', val_chamfer_loss, step)
            print('[%d: %d/%d] val loss: %f' % (epoch, j, val_num_batch, val_chamfer_loss.item()))

            early_stopping(val_chamfer_loss, net)
            if early_stopping.early_stop:
                print("Early Stopping: %d" % epoch)

                total_correct = 0
                total_testset = 0
                vis = visdom.Visdom()
                for i, data in tqdm(enumerate(test_dataloader, 0)):
                    points, _ = data
                    points = points.numpy().transpose(0, 2, 1)
                    points = torch.from_numpy(points).float()
                    target = points
                    points, target = points.to(device), target.to(device)
                    net.eval()
                    with torch.no_grad():
                        gen = net(points)

                    gen = torch.transpose(gen, 1, 2)
                    target = torch.transpose(target, 1, 2)
                    test_chamfer_loss, _ = chamfer_distance(gen, target)
                    total_test_loss += test_chamfer_loss

                    vis.scatter(target[i])
                    vis.scatter(gen[i])

                total_test_loss = total_test_loss / float(test_num_batch)
                print('[%d: ] best test loss: %f' % (epoch, total_test_loss.item()))
                break
        if early_stopping.early_stop:
            break
    if early_stopping.early_stop:
        break

torch.save(net, epoch + 1,  '%s/cls_last_model_%d.pth' % (args.outf, epoch))
writer_train.close()
writer_val.close()




