import tensorflow as tf
import os
import sys
import open3d as o3d
import numpy as np
import h5py
import matplotlib as plt
from matplotlib import cm
def load_data(h5_filename, mode):
    if mode == 'train':
        data = np.empty((5000, 2048, 3))
        label = np.empty((5000))
        with h5py.File(h5_filename) as f:
            for i in range(5000):
                d = f[str(i)]
                idxs = np.arange(0, d["points"][:].shape[0])
                np.random.shuffle(idxs)
                data[i, :, :] = d["points"][:][idxs[:2048]]
                label[i] = int(d.attrs["label"])
    elif mode =='test':
        data = np.empty((1000, 2048, 3))
        label = np.empty((1000))
        with h5py.File(h5_filename) as f:
            for i in range(1000):
                d = f[str(i)]
                idxs = np.arange(0, d["points"][:].shape[0])
                np.random.shuffle(idxs)
                data[i, :, :] = d["points"][:][idxs[:2048]]
                label[i] = int(d.attrs["label"])
    return data, label

# Translate data to color
def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)  #ScalarMappable:Use data normalization before returning RGBA colors from the given color map
    return s_m.to_rgba(array)[:,:-1]        #to_rgba:Return a normalized rgba array corresponding to *x*

def translate(x):
    xx = np.ndarray((x.shape[0], 4096, 3))
    for i in range(x.shape[0]):
        xx[i] = array_to_color(x[i])
    # Free Memory
    del x

    return xx

def plot_pc(sample_pcs, mode, visualization=False):
    # Visualize the demo
    pcl = o3d.geometry.PointCloud()
    show_pcl = o3d.geometry.PointCloud()
    all_pcls = []
    for i in range(int(sample_pcs.shape[0])):
        print("Visualizing: %03d/%03d" % (i, sample_pcs.shape[0]))
        pts = sample_pcs[i].reshape(-1, 3)
        pcl.points = o3d.utility.Vector3dVector(pts)    #Convert float64 numpy array of shape (n, 3) to Open3D format.
        if mode == "estimate normals":
            pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
            show_pcl = pcl
        elif mode == "plot_voxel_grid":
            num_pts = sample_pcs[i].shape[0]
            pcl.scale(1 / np.max(pcl.get_max_bound() - pcl.get_min_bound()), center=pcl.get_center())       #scale to a unit cube
            pcl.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size = (num_pts, 3)))
            pcl_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcl, voxel_size=0.05)
            show_pcl = pcl_voxel
        else:
            show_pcl = pcl

        if visualization == True:
            o3d.visualization.draw_geometries([show_pcl])        #draw geometries: function to visualize point clouds

        all_pcls.append(show_pcl)

    return all_pcls

def load_data_voxel(all_data_dir):
    with h5py.File(all_data_dir, 'r') as h5:
        train_data, train_label = h5["X_train"][:], h5["y_train"][:]
        test_data, test_label = h5["X_test"][:], h5["y_test"][:]
    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
    return train_data,train_label, test_data, test_label

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


if __name__=="__main__":
    
    train_data_dir = os.path.join(data_dir, "train_point_clouds.h5")
    #test_data_dir = os.path.join(data_dir, "test_point_clouds.h5")
    train_data, train_label = load_data(train_data_dir, 'train')
    print(train_data.shape)
    print(train_label.shape)
    #test_data, test_label = load_data(test_data_dir, 'test')

    mode = "plot_voxel_grid"
    data = train_data[:5, :,:]
    all_pcls = plot_pc(data, mode, False)

    current_data = train_data[0, :, :]
    print(current_data.shape)
    Voxels = voxelgrid.VoxelGrid(current_data, x_y_z=[32, 32, 32], bb_cuboid=True, build=True)
    print('n_voxels: ', Voxels.n_voxels)
    print('xyzmax: ', Voxels.xyzmax)
    print('xyzmin: ', Voxels.xyzmin)
    print(Voxels.shape)
    #process_voxel_input(all_pcls[0])
    

    '''
    all_data_dir = os.path.join(data_dir, "full_dataset_vectors.h5")
    train_data, train_label, test_data, test_label = load_data_voxel(all_data_dir)
    train_label = to_categorical(train_label, num_classes=10)
    # y_test = to_categorical(y_test, num_classes=10)

    train_data = translate(train_data).reshape(-1, 16, 16, 16, 3)
    #test_data = translate(test_data).reshape(-1, 16, 16, 16, 3)
    print(train_data.shape, train_label.shape)
    '''



