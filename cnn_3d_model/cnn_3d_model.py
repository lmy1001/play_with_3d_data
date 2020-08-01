import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import random

def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CNN_3d_Model(nn.Module):
    def __init__(self):
        super(CNN_3d_Model, self).__init__()
        self.conv1 = nn.Conv3d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool3d(2)

        self.bn1 = nn.BatchNorm3d(8)
        self.bn2 = nn.BatchNorm3d(16)
        self.bn3 = nn.BatchNorm3d(32)
        self.bn4 = nn.BatchNorm3d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 10)

        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.bn_fc2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))                         #size:(8, 16, 16, 16)
        x = F.relu(self.bn2(self.conv2(x)))                         #size:(16, 16, 16, 16)
        x = self.pool(x)                                            #size:(16, 8, 8, 8)

        x = F.relu(self.bn3(self.conv3(x)))                         #size: (32, 8, 8, 8)
        x = F.relu(self.bn4(self.conv4(x)))                         #size: (64, 8, 8, 8)
        x = self.pool(x)                                            #size: (64, 4, 4, 4)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 4 * 4 *4)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.log_softmax(self.fc3(x), dim=1)

        return x

if __name__=='__main__':
    sim_data = Variable(torch.rand(3, 3, 16, 16, 16))
    #sim_label = Variable([0, 1, 2])
    cls = CNN_3d_Model()
    out = cls(sim_data)
    print(out)

    out_pred = np.argmax(out, axis=1)
    print(out_pred)