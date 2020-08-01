import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import os
import argparse
import random
import numpy as np
from tensorboardX import SummaryWriter
from pytorchtools import EarlyStopping
import cnn_3d_model
import data_load

#parsing and configuration
def parse_args():
    desc = "tensorflow implementation of 3d cnn classifer in mnist3d dataset"
    parser = argparse.ArgumentParser(desc)

    parser.add_argument('--device', type=str, default='cpu',
                        help='if choose cpu, only test with subset of data, else test with whole data')
    parser.add_argument('--data_dir', type=str,
                        default='/home/menliu/Dataset/mnist_3d/',
                        #default='/Users/lmy/Dataset/mnist_3d/',
                        help='directory of dataset')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint/best_model.h5',
                        help='directory to store best checkpoint')
    parser.add_argument('--use_batch_norm', type=bool, default=True,
                        help='Boolean for using batch normalization')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer to choose', choices=['Adam', 'SGD'])
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=250, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for initializing training. ')
    parser.add_argument('--train_log_name', type=str, default='log/train_log',
                        help='File name of train log event')
    parser.add_argument('--val_log_name', type=str, default='log/val_log',
                        help='File name of validation log event')
    return parser.parse_args()


def main(args):
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    all_data_dir = os.path.join(args.data_dir,"full_dataset_vectors.h5")

    #load data
    train_data_ori, train_label_ori, test_data, test_label = data_load.load_data_voxel(all_data_dir)

    if args.device == 'cpu':
        train_data = train_data_ori[150:1000, :]
        train_label = train_label_ori[150:1000]
        val_data = train_data_ori[:150, :]
        val_label = train_label_ori[:150]
        test_data = test_data[:100, :]
        test_label = test_label[:100]
    else:
        train_data = train_data_ori[1500:, :]
        train_label = train_label_ori[1500:]
        val_data = train_data_ori[:1500, :]
        val_label = train_label_ori[:1500]


    train_data = data_load.translate(train_data).reshape(-1, 16, 16, 16, 3)
    train_data = train_data.transpose(0, 4, 1, 2, 3)
    test_data = data_load.translate(test_data).reshape(-1, 16, 16, 16, 3)
    test_data = test_data.transpose(0, 4, 1, 2, 3)
    val_data = data_load.translate(val_data).reshape(-1, 16, 16, 16, 3)
    val_data = val_data.transpose(0, 4, 1, 2, 3)

    #create network
    cls = cnn_3d_model.CNN_3d_Model().to(device)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(list(cls.parameters()), lr=args.learn_rate)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(list(cls.parameters()), lr=args.learn_rate, momentum=args.momentum)
    else:
        print("wrong optimizer")
        return

    #init weights
    if args.seed is None:
        args.seed = random.randint(0, 10000)
    cnn_3d_model.set_random_seed(args.seed)

    # output log to writer
    writer_train = SummaryWriter(logdir=args.train_log_name)
    writer_val = SummaryWriter(logdir=args.val_log_name)

    #training
    batch_size = args.batch_size
    epochs = args.num_epochs
    num_data = train_data.shape[0]
    num_batches = int(num_data / batch_size)
    num_val_batches = int(val_data.shape[0] / batch_size)

    test_data, test_label = torch.from_numpy(test_data).float().to(device), \
                            torch.from_numpy(test_label).to(device)

    for epoch in range(epochs):
        train_data, train_label, _ = data_load.shuffle_data(train_data, train_label)

        cls.train()
        train_loss = 0
        tot_val_loss = 0
        for i in range(num_batches):
            idx = (i * batch_size) % num_data
            end_idx = idx + batch_size
            batch_input = train_data[idx:end_idx, :, :, :, :]
            batch_target = train_label[idx:end_idx]

            batch_input, batch_target = torch.from_numpy(batch_input).float().to(device), \
                                        torch.from_numpy(batch_target).to(device)

            #train
            optimizer.zero_grad()
            pred = cls(batch_input)
            loss = F.nll_loss(pred, batch_target)
            train_loss = train_loss + loss.item()

            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(batch_target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % \
                  (epoch, i,num_batches, loss.item(), correct.item() / float(batch_size)))

        train_loss = train_loss / float(num_batches)
        writer_train.add_scalar('loss', train_loss, epoch)

        #evaluate
        cls.eval()
        for j in range(num_val_batches):
            idx = (j * batch_size) % (val_data.shape[0])
            end_idx = idx + batch_size
            batch_input = val_data[idx:end_idx, :, :, :, :]
            batch_target = val_label[idx:end_idx]

            batch_input, batch_target = torch.from_numpy(batch_input).float().to(device), \
                                        torch.from_numpy(batch_target).to(device)

            pred_val = cls(batch_input)
            val_loss = F.nll_loss(pred_val, batch_target)
            tot_val_loss = tot_val_loss + val_loss

            pred_choice_val = pred_val.data.max(1)[1]
            val_correct = pred_choice_val.eq(batch_target.data).cpu().sum()
            print('[%d: ] val loss: %f accuracy: %f' % \
                  (epoch, val_loss.item(), val_correct.item() / float(batch_size)))

        tot_val_loss = tot_val_loss / float(num_val_batches)
        writer_val.add_scalar('loss', tot_val_loss, epoch)

        early_stopping(tot_val_loss, cls)
        if early_stopping.early_stop:
            print("Early Stopping: %d" % epoch)

            cls.eval()
            pred_test = cls(test_data)
            test_loss = F.nll_loss(pred_test, test_label)
            pred_choice_test = pred_test.data.max(1)[1]
            test_correct = pred_choice_test.eq(test_label.data).cpu().sum()
            print('[%d: ] test loss: %f accuracy: %f' % \
                (epoch, test_loss.item(), test_correct.item() / float(test_data.shape[0])))
            break

    torch.save(cls.state_dict(), './saved_model/cls_model_last_epoch_%d.pt' % epoch)
    writer_train.close()
    writer_val.close()

if __name__=='__main__':
    args = parse_args()

    main(args)