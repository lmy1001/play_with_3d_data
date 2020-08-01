from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import numpy as np
import h5py

class Mnist3dDataset(data.Dataset):
    def __init__(self, root,
                 npoints=2048,
                 split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.meta = {}
        if self.split == 'train':
            filename = os.path.join(self.root, "train_point_clouds.h5")
            num = 5000
        else:
            filename = os.path.join(self.root, "test_point_clouds.h5")
            num = 1000

        with h5py.File(filename, 'r') as f:
            for i in range(num):
                d = f[str(i)]
                idxs = np.arange(0, d["points"][:].shape[0])
                np.random.shuffle(idxs)
                data = d["points"][:][idxs[:self.npoints]]
                label = int(d.attrs["label"])
                self.meta[i] = [data, label]

    def __getitem__(self, index):
        points = self.meta[index][0]
        label = self.meta[index][1]

        return points, label

    def __len__(self):
        return len(self.meta)




