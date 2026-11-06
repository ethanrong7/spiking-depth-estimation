import numpy as np
import torch
import torch.utils.data as data
import logging
import copy
import random
from glob import glob
import os.path as osp
from prefetch_generator import BackgroundGenerator
from core.utils.reader import *



class Gs1bSynthSpikeDataset(data.Dataset):
    def __init__(self, split='train'):
        self.is_test = False
        self.init_seed = False
        self.depth_list = []
        self.spike_list = []
        self.extra_info = []

        root = '/data2/gaozhuoheng/RESULTS_gao/gs1b-synth/scenes_gao_train'
        # root = '/data2/gaozhuoheng/RESULTS_gao/gs1b-synth/scenes_gao_test'

        spike1_list = sorted( glob(osp.join(root, f'*/synthetic/spike_l/*.dat')) )
        spike2_list = sorted( glob(osp.join(root, f'*/synthetic/spike_r/*.dat')) )
        depth_list = sorted( glob(osp.join(root, f'*/synthetic/depth/*.exr')))
        
        np.random.seed(42)
        if split != 'train':
            length = len(spike1_list)
            # Generate random indices for 10% of the dataset

            idx = np.random.choice(length, 30, replace=False)
            # idx = list(range(length))
            spike1_list = [spike1_list[i] for i in idx]
            spike2_list = [spike2_list[i] for i in idx]
            depth_list = [depth_list[i] for i in idx]
            
        for spike1, spike2, depth in zip(spike1_list, spike2_list, depth_list):
            self.spike_list += [ [spike1, spike2] ]
            self.depth_list += [ depth ]
        
    def __getitem__(self, index):    
        if self.is_test:
            spike1 = get_block_spikes(self.spike_list[index][0]).astype(int)
            spike2 = get_block_spikes(self.spike_list[index][1]).astype(int)
            spike1 = torch.from_numpy(spike1).float()
            spike2 = torch.from_numpy(spike2).float()
            return spike1, spike2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.spike_list)
        depth = read_depth(self.depth_list[index])
        disp = depth2disp(depth, f=289.740625, B=0.08)
        
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        spike1 = get_block_spikes(self.spike_list[index][0]).astype(int)
        spike2 = get_block_spikes(self.spike_list[index][1]).astype(int)
        
        disp = np.array(-disp).astype(np.float32)
        
        spike1 = torch.from_numpy(spike1).float()
        spike2 = torch.from_numpy(spike2).float()
        disp = torch.from_numpy(disp).unsqueeze(0)
        valid = torch.from_numpy(valid).float()

        return self.spike_list[index] + [self.depth_list[index]], spike1, spike2, disp, valid


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.spike_list = v * copy_of_self.spike_list
        copy_of_self.depth_list = v * copy_of_self.depth_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.spike_list)
    
    
class RealSpikeDataset(data.Dataset):
    def __init__(self, split='train'):
        self.is_test = False
        self.init_seed = False
        self.depth_list = []
        self.spike_list = []
        self.extra_info = []

        root = '/home/ifgovh/New_visuo/RAFT-Stereo-spike_real/real_data/camera'

        spike1_list = sorted( glob(osp.join(root, 'left/**/*.dat')) )
        spike2_list = sorted( glob(osp.join(root, 'right/**/*.dat')) )
        depth_list = sorted( glob(osp.join(root, 'depth/**/*.npy')) )

        np.random.seed(42)
        if split != 'train':
            length = len(spike1_list)
            idx = np.random.choice(length, 100, replace=False)

            # Select the corresponding subsets of spikes and disparity maps
            spike1_list = [spike1_list[i] for i in idx]
            spike2_list = [spike2_list[i] for i in idx]
            depth_list = [depth_list[i] for i in idx]
            
        for spike1, spike2, depth in zip(spike1_list, spike2_list, depth_list):
            self.spike_list += [ [spike1, spike2] ]
            self.depth_list += [ depth ]
        
    def __getitem__(self, index):
        if self.is_test:
            spike1 = get_spike_matrix(self.spike_list[index][0], Time_head=True).astype(int)[50:100, :, :400]
            spike2 = get_spike_matrix(self.spike_list[index][1], Time_head=True).astype(int)[50:100, :, :400]
            spike1 = torch.from_numpy(spike1).float()
            spike2 = torch.from_numpy(spike2).float()
            return spike1, spike2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.spike_list)
        depth = np.load(self.depth_list[index])
        disp = depth2disp(0.001 * depth, f=289.740625, B=0.08)  # camera
        
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512
        
        spike1 = get_spike_matrix(self.spike_list[index][0], Time_head=True).astype(int)[50:100, :, :400]
        spike2 = get_spike_matrix(self.spike_list[index][1], Time_head=True).astype(int)[50:100, :, :400]
        
        disp = np.array(-disp).astype(np.float32)
        
        spike1 = torch.from_numpy(spike1).float()
        spike2 = torch.from_numpy(spike2).float()
        disp = torch.from_numpy(disp).unsqueeze(0)
        valid = torch.from_numpy(valid).float()
        
        return self.spike_list[index] + [self.disparity_list[index]], spike1, spike2, disp, valid

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.spike_list = v * copy_of_self.spike_list
        copy_of_self.depth_list = v * copy_of_self.depth_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.spike_list)
    

def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name == 'gs1b-synth-spike':
            new_dataset = Gs1bSynthSpikeDataset()
        elif dataset_name == 'real-spike':
            new_dataset = RealSpikeDataset()

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = DataLoaderX(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=6, drop_last=True)   

    logging.info('Training with %d spike pairs' % len(train_dataset))
    return train_loader


class DataLoaderX(data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
