import torch
import os
import sys
import numpy as np
import random
import scipy.io as sio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from einops import rearrange
import copy
import gc

import scipy
from sea import StableEA, compute_R_ref, EA

from torch.utils.data import Dataset,DataLoader

from scipy import stats

# from Modules.spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified,SPDVectorize
# from Data_process.process_function import Load_BCIC_2a_raw_data
from collections import Counter
current_path = os.path.abspath('./')
root_path = current_path # os.path.split(current_path)[0]

sys.path.append(root_path)

        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def select_devices(num_device,gpus=None):
    if gpus is None:
        gpus = torch.cuda.device_count()  
        gpus = [i for i in range(gpus)]
        
    res = []
    last_id = 0
            
    min_memory = 25447170048 // 2  
    for i in range(num_device):
        device_id = gpus[last_id%len(gpus)]
        last_id+=1
        while torch.cuda.get_device_properties(device_id).total_memory < min_memory:
            device_id = gpus[last_id%len(gpus)]
            last_id+=1
        res.append(torch.device(f'cuda:{device_id}') )
    return res

def select_free_gpu():  

    gpus = torch.cuda.device_count()  
    if gpus == 0:  
        return None  
    else:  
        device_id = 0  
        min_memory = 25447170048 // 2  
        while True:
            i = random.randint(0, gpus-1)
        # for i in range(gpus):  
            mem_info = torch.cuda.get_device_properties(i)  
            # print(mem_info.total_memory)
            if mem_info.total_memory > min_memory:  
                device_id = i  
                break

        return torch.device(f'cuda:{device_id}') 

def rand_mask(feature):

    for _ in range(np.random.randint(0,4)):
        c = np.random.randint(0,22)

        a = np.random.normal(1,0.4,1)[0]

        feature[:,c] *=a
    return feature

def rand_cov(x):
    # print('xt shape:',xt.shape)
    E = torch.matmul(x, x.transpose(1,2))
    # print(E.shape)
    R = E.mean(0)
    
    U, S, V = torch.svd(R)
    R_mat = U@torch.diag(torch.rand(S.shape[0])*2)@V
    new_x = torch.einsum('n c s,r c -> n r s',x,R_mat)
    return new_x


def shuffle_data(dataset):
    x = rearrange(dataset.x,'(n i) c s->n i c s',n=16)
    y = rearrange(dataset.y,'(n i)->n i',n=16)
    new_x = []
    new_y = []

    for i in np.random.permutation(x.shape[0]):
        index = np.random.permutation(x.shape[1])
        new_x.append(x[i][index])
        new_y.append(y[i][index])

    new_x = torch.stack(new_x)
    new_y = torch.stack(new_y)
    new_x = rearrange(new_x,'a b c d->(a b) c d')
    new_y = rearrange(new_y,'a b->(a b)')

    return eeg_dataset(new_x,new_y)


def print_log(s,path="log.txt"):
    with open(path,"a+") as f:
        f.write((str(s) if type(s) is not str else s) +"\n")
def callback(res):
        print('<进程%s> subject %s accu %s' %(os.getpid(),res['sub'], str(res["accu"])))
        
        
def geban(batch_size=10, n_class=4):
    res = [random.randint(0, batch_size) for i in range(n_class-1) ]
    res.sort()
    # print(res)
    ret=[]
    last=0
    for r in res:
        ret.append(r-last)
        last=r
    ret.append(batch_size-last)
    return ret

def geban_entropy(batch_size=10, n_class=4, entropy_scope=[0,1]):
    while True:
        num_class = geban(batch_size, n_class)
        total = sum(num_class)
        ent = stats.entropy([x/total for x in num_class], base=n_class)
        if entropy_scope[0]<=ent and ent<=entropy_scope[1]: break
    return num_class

def sample(batch_size=10, n_class=4):
    res = [random.randint(0, n_class-1) for i in range(batch_size) ]
    res = Counter(res)
    ret = []
    for i in range(n_class):
        ret.append(res[i])
    return ret

def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

# Build a Dataset class for reading validation and test set data.
class eeg_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None):
        super(eeg_dataset,self).__init__()

        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y        

    
def get_subj_data(sub, data_path, few_shot_number = 1, is_few_EA = False, target_sample=-1, sess=None, use_average=False):
    
    # target_y_data = []
    
    i=sub
    R=None
    source_train_x = []
    source_train_y = []
    source_valid_x = []
    source_valid_y = []
    
    if sess is not None:
        
        train_path = os.path.join(data_path,r'sub{}_sess{}_train/Data.mat'.format(i, sess))
        test_path = os.path.join(data_path,r'sub{}_sess{}_test/Data.mat'.format(i, sess))
    else:
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        
    train_data = sio.loadmat(train_path)
    test_data = sio.loadmat(test_path)
    if use_average:
        train_data['x_data'] = train_data['x_data'] - train_data['x_data'].mean(-2, keepdims=True)
    if is_few_EA is True:
        session_1_x = EA(train_data['x_data'],R)
    else:
        session_1_x = train_data['x_data']

    session_1_y = train_data['y_data'].reshape(-1)

    train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1,stratify = session_1_y)
    
    source_train_x.extend(train_x)
    source_train_y.extend(train_y)
    
    source_valid_x.extend(valid_x)
    source_valid_y.extend(valid_y)
    if use_average:
        test_data['x_data'] = test_data['x_data'] - test_data['x_data'].mean(-2, keepdims=True)
        
    if is_few_EA is True:
        session_2_x = EA(test_data['x_data'],R)
    else:
        session_2_x = test_data['x_data']

    session_2_y = test_data['y_data'].reshape(-1)

    train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.1,stratify = session_2_y)
    
    source_train_x.extend(train_x)
    source_train_y.extend(train_y)
    
    source_valid_x.extend(valid_x)
    source_valid_y.extend(valid_y)
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y)
    valid_datset  = eeg_dataset(source_valid_x,source_valid_y)
    
    return train_dataset, valid_datset

import glob
def get_seed4_data(target_sub, data_path, valid_ratio=0.1, is_few_EA=True, use_channels=None):
    """
    LOSO-style split:
    - target_sub: test subject id (int)
    - data_path: directory containing 1_i.mat
    """
    original_channels = [
        # ===== Frontal pole / Anterior frontal =====
        'FP1', 'FPZ', 'FP2',
        'AF3',        'AF4',

        # ===== Frontal =====
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',

        # ===== Fronto-temporal / Fronto-central =====
        'FT7',
        'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
        'FT8',

        # ===== Temporal / Central =====
        'T7',
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
        'T8',

        # ===== Temporo-parietal / Centro-parietal =====
        'TP7',
        'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
        'TP8',

        # ===== Parietal =====
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',

        # ===== Parieto-occipital =====
        'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',

        # ===== Occipital / Cerebellar =====
        'CB1',
        'O1', 'OZ', 'O2',
        'CB2',
    ]

    train_data_list = []
    train_label_list = []

    test_data = None
    test_label = None

    R = None

    # ---- 1. Scan all participant files. ----
    mat_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))

    if len(mat_files) == 0:
        raise RuntimeError(f"No .mat files found in {data_path}")

    for mat_path in mat_files:
        name = os.path.basename(mat_path)
        sub_id = int(os.path.splitext(name)[0].split("_")[1])

        mat = sio.loadmat(mat_path)
        data = mat["data"]    # (N, 62, 1024)
        if is_few_EA:
            data = EA(data, R)
        label = mat["label"].squeeze()  # (N,)

        if sub_id == target_sub:
            test_data = data
            if is_few_EA:
                test_data = EA(test_data, R)
            test_label = label
        else:
            train_data_list.append(data)
            train_label_list.append(label)

    if test_data is None:
        raise ValueError(f"Target subject {target_sub} not found")

    # ---- 2. Combine training subjects ----
    train_data_all = np.concatenate(train_data_list, axis=0)
    train_label_all = np.concatenate(train_label_list, axis=0)

    # Add here: If use_channels is not None, select the channels from original_channels that are present in use_channels,
    # and modify train_data_all and test_data accordingly.
    if use_channels is not None:
        channel_indices = [original_channels.index(ch) for ch in use_channels if ch in original_channels]
        train_data_all = train_data_all[:, channel_indices, :]
        test_data = test_data[:, channel_indices, :]

    # ---- 3. Splitting the validation set from the training set. ----
    if valid_ratio > 0.0:
        X_train, X_valid, y_train, y_valid = train_test_split(
            train_data_all,
            train_label_all,
            test_size=valid_ratio,
            stratify=train_label_all
        )
    else:
        X_train = train_data_all
        X_valid = np.array([])
        y_train = train_label_all
        y_valid = np.array([])

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train).view(-1)

    test_data  = torch.FloatTensor(test_data)
    test_label  = torch.LongTensor(test_label).view(-1)

    if X_valid is not None:
        X_valid = torch.FloatTensor(X_valid)
        y_valid = torch.LongTensor(y_valid).view(-1)

    # ---- 4. Build Dataset ----
    train_dataset = eeg_dataset(X_train, y_train)
    print(X_train.shape)
    if len(X_valid) > 0:
        valid_dataset = eeg_dataset(X_valid, y_valid)
    else :
        valid_dataset = None
    test_dataset  = eeg_dataset(test_data, test_label)
    print(test_data.shape)

    return train_dataset, valid_dataset, test_dataset

def get_data(sub,data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1, use_avg=True, use_channels=None):
    # 1) First, calculate the global reference (using only the source training data).
    # R_ref = None
    # if is_few_EA:
    #     R_ref = compute_R_ref(data_path, target_sub=sub)
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
        
    if is_few_EA is True:
        session_2_x = EA(session_2_data['x_data'],R)
    else:
        session_2_x = session_2_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample, use_avg=use_avg)
        test_x_2 = temporal_interpolation(test_x_2, target_sample, use_avg=use_avg)
    if use_channels is not None:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0)[:,use_channels,:],torch.cat([test_y_1,test_y_2],dim=0))
    else:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2],dim=0))

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in range(1,10):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)

        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'],R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.1,stratify = session_2_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample, use_avg=use_avg)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample, use_avg=use_avg)
        
    if use_channels is not None:
        train_dataset = eeg_dataset(source_train_x[:,use_channels,:],source_train_y,source_train_s)
    else:
        train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    if use_channels is not None:
        valid_datset = eeg_dataset(source_valid_x[:,use_channels,:],source_valid_y,source_valid_s)
    else:
        valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset

def get_default_10_20_32ch_names():
    ch32 = [
        "Fp1", "Fp2",
        "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6",
        "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6",
        "P7", "P3", "Pz", "P4", "P8",
        "O1", "Oz", "O2",
        "A1", "A2",   # 常见的参考（或耳垂/乳突）
        "AFz"         # 有些帽子会有 AFz/POz 等（这里占位为 32）
    ]
    return ch32

def get_default_30ch_names():
    ch32 = get_default_10_20_32ch_names()
    DROP_CHS = {"A1", "A2"}
    ch30 = [c for c in ch32 if c not in DROP_CHS]

    if len(ch30) != 30:
        ch30 = ch32[:30]
    return ch30

def _normalize_name(name: str) -> str:
    name = name.strip()
    alias = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
        "Pz": "PZ",
        "Fz": "FZ",
        "Cz": "CZ",
        "Oz": "OZ",
    }
    std = name[0].upper() + name[1:]
    std = std.replace("z", "Z")
    return alias.get(std, std)

def print_label_stats(y, name):
    labels, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    print(f"\n{name} label distribution:")
    for l, c in zip(labels, counts):
        print(f"  label {l}: {c} ({c/total:.2%})")
    print(f"  total: {total}")

def get_wmsub_data(data_path, sub, use_channels=None, seed=2024, split=(0.6, 0.2, 0.2)):
    ch_names_30 = [_normalize_name(c) for c in get_default_30ch_names()]
    if use_channels is not None:
        use_channels_norm = [_normalize_name(c) for c in use_channels]
        name_to_idx = {c: i for i, c in enumerate(ch_names_30)}

        missing = [c for c in use_channels_norm if c not in name_to_idx]
        if missing:
            raise ValueError(
                f"use_channels 中有通道不在默认 30 通道列表里：{missing}\n"
                f"默认通道列表为：{ch_names_30}"
            )
        pick_idx = [name_to_idx[c] for c in use_channels_norm]
        picked_names = [ch_names_30[i] for i in pick_idx]
    else:
        pick_idx = None
        picked_names = ch_names_30
    fp = os.path.join(data_path, f"Sub_S1_{sub}.mat")
    mat = sio.loadmat(fp)
    data = mat["data1"]
    label = mat["label1"]
    label = np.asarray(label).reshape(-1)
    label_map = {11: 0, 14: 1}
    label = np.vectorize(label_map.get)(label)
    if pick_idx is not None:
        data = data[pick_idx, :, :]  # [C_sel, T, N]
    data = np.transpose(data, (2, 0, 1))  # [N, C, T]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(label))
    rng.shuffle(idx)
    n_total = len(idx)
    n_train = int(round(split[0] * n_total))
    n_valid = int(round(split[1] * n_total))

    n_test = n_total - n_train - n_valid

    idx_train = idx[:n_train]
    idx_valid = idx[n_train:n_train + n_valid]
    idx_test = idx[n_train + n_valid:]

    X_train, y_train = data[idx_train], label[idx_train]
    X_valid, y_valid = data[idx_valid], label[idx_valid]
    X_test, y_test = data[idx_test], label[idx_test]

    print_label_stats(y_train, "Train")
    print_label_stats(y_valid, "Valid")
    print_label_stats(y_test,  "Test")

    train_dataset = eeg_dataset(X_train, y_train)
    valid_dataset = eeg_dataset(X_valid, y_valid)
    test_dataset = eeg_dataset(X_test, y_test)

    return train_dataset, valid_dataset, test_dataset

def get_wm_data(data_path, use_channels=None, seed=2025, split=(0.6, 0.2, 0.2)):
    assert abs(sum(split) - 1.0) < 1e-6

    mat_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))
    if len(mat_files) == 0:
        raise FileNotFoundError(f"在 {data_path} 下没有找到任何 .mat 文件")

    # Default 30 channel names (you can change them according to your actual equipment)
    ch_names_30 = [_normalize_name(c) for c in get_default_30ch_names()]

    if use_channels is not None:
        use_channels_norm = [_normalize_name(c) for c in use_channels]
        name_to_idx = {c: i for i, c in enumerate(ch_names_30)}

        missing = [c for c in use_channels_norm if c not in name_to_idx]
        if missing:
            raise ValueError(
                f"use_channels 中有通道不在默认 30 通道列表里：{missing}\n"
                f"默认通道列表为：{ch_names_30}"
            )
        pick_idx = [name_to_idx[c] for c in use_channels_norm]
        picked_names = [ch_names_30[i] for i in pick_idx]
    else:
        pick_idx = None
        picked_names = ch_names_30

    X_list, y_list = [], []

    for fp in mat_files:
        mat = sio.loadmat(fp)
        if "data1" not in mat or "label1" not in mat:
            raise KeyError(f"{os.path.basename(fp)} 缺少 data1 或 label1")

        data = mat["data1"]
        label = mat["label1"]

        # label: [N,1] 或 [N,]
        label = np.asarray(label).reshape(-1)
        label_map = {11: 0, 14: 1}
        label = np.vectorize(label_map.get)(label)

        if pick_idx is not None:
            data = data[pick_idx, :, :]  # [C_sel, T, N]

        data = np.transpose(data, (2, 0, 1))  # [N, C, T]

        X_list.append(data.astype(np.float32))
        y_list.append(label.astype(np.int64))

    X = np.concatenate(X_list, axis=0)  # [N_total, C, T]
    y = np.concatenate(y_list, axis=0)  # [N_total]

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    n_total = len(idx)
    n_train = int(round(split[0] * n_total))
    n_valid = int(round(split[1] * n_total))

    n_test = n_total - n_train - n_valid

    idx_train = idx[:n_train]
    idx_valid = idx[n_train:n_train + n_valid]
    idx_test = idx[n_train + n_valid:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_valid, y_valid = X[idx_valid], y[idx_valid]
    X_test, y_test = X[idx_test], y[idx_test]
    print_label_stats(y_train, "Train")
    print_label_stats(y_valid, "Valid")
    print_label_stats(y_test,  "Test")

    info = {
        "total_samples": int(n_total),
        "train_samples": int(n_train),
        "valid_samples": int(n_valid),
        "test_samples": int(n_test),
        "channels": picked_names,
        "data_shape": {"X": list(X.shape), "y": list(y.shape)},
        "split": split,
        "seed": seed,
        "files": [os.path.basename(p) for p in mat_files],
    }
    print(info)

    train_dataset = eeg_dataset(X_train, y_train)
    valid_dataset = eeg_dataset(X_valid, y_valid)
    test_dataset = eeg_dataset(X_test, y_test)

    return train_dataset, valid_dataset, test_dataset

def get_wm_data_per_subject_split(data_path, use_channels=None, seed=2025, split=(0.6, 0.2, 0.2)):
    assert abs(sum(split) - 1.0) < 1e-6

    mat_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))
    if len(mat_files) == 0:
        raise FileNotFoundError(f"在 {data_path} 下没有找到任何 .mat 文件")

    ch_names_30 = [_normalize_name(c) for c in get_default_30ch_names()]

    if use_channels is not None:
        use_channels_norm = [_normalize_name(c) for c in use_channels]
        name_to_idx = {c: i for i, c in enumerate(ch_names_30)}
        missing = [c for c in use_channels_norm if c not in name_to_idx]
        if missing:
            raise ValueError(f"use_channels 中有通道不在默认 30 通道列表里：{missing}\n默认通道列表为：{ch_names_30}")
        pick_idx = [name_to_idx[c] for c in use_channels_norm]
        picked_names = [ch_names_30[i] for i in pick_idx]
    else:
        pick_idx = None
        picked_names = ch_names_30

    X_tr_list, y_tr_list = [], []
    X_va_list, y_va_list = [], []
    X_te_list, y_te_list = [], []

    label_map = {11: 0, 14: 1}

    for fp in mat_files:
        mat = sio.loadmat(fp)
        if "data1" not in mat or "label1" not in mat:
            raise KeyError(f"{os.path.basename(fp)} 缺少 data1 或 label1")

        data = mat["data1"]          # [30, 500, 240]
        label = mat["label1"]        # [240, 1] 或 [240]

        if data.ndim != 3:
            raise ValueError(f"{os.path.basename(fp)} 的 data1 维度应为 3，实际为 {data.shape}")

        C, T, N = data.shape
        if C != 30:
            raise ValueError(f"{os.path.basename(fp)} 的 data1 第一维应为 30 通道，实际为 {C}")

        label = np.asarray(label).reshape(-1).astype(int)

        label = np.vectorize(label_map.get)(label).astype(np.int64)

        if label.shape[0] != N:
            raise ValueError(f"{os.path.basename(fp)} label1 样本数 {label.shape[0]} 与 data1 样本数 {N} 不一致")

        if pick_idx is not None:
            data = data[pick_idx, :, :]   # [C_sel, T, N]

        data = np.transpose(data, (2, 0, 1)).astype(np.float32)

        sub_seed = (abs(hash(os.path.basename(fp))) + seed) % (2**32)
        rng = np.random.default_rng(sub_seed)

        idx = np.arange(N)
        rng.shuffle(idx)

        n_train = int(round(split[0] * N))
        n_valid = int(round(split[1] * N))
        n_test = N - n_train - n_valid

        idx_tr = idx[:n_train]
        idx_va = idx[n_train:n_train + n_valid]
        idx_te = idx[n_train + n_valid:]

        X_tr_list.append(data[idx_tr]); y_tr_list.append(label[idx_tr])
        X_va_list.append(data[idx_va]); y_va_list.append(label[idx_va])
        X_te_list.append(data[idx_te]); y_te_list.append(label[idx_te])

    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    X_valid = np.concatenate(X_va_list, axis=0)
    y_valid = np.concatenate(y_va_list, axis=0)
    X_test  = np.concatenate(X_te_list, axis=0)
    y_test  = np.concatenate(y_te_list, axis=0)

    print_label_stats(y_train, "Train")
    print_label_stats(y_valid, "Valid")
    print_label_stats(y_test,  "Test")

    info = {
        "channels": picked_names,
        "split": split,
        "seed": seed,
        "files": [os.path.basename(p) for p in mat_files],
        "train_samples": int(len(y_train)),
        "valid_samples": int(len(y_valid)),
        "test_samples": int(len(y_test)),
    }
    print(info)

    train_dataset = eeg_dataset(X_train, y_train)
    valid_dataset = eeg_dataset(X_valid, y_valid)
    test_dataset  = eeg_dataset(X_test,  y_test)

    return train_dataset, valid_dataset, test_dataset

def get_data_openbmi(data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1):
    test_rate = 0.5
    subs_list = np.int32(np.linspace(1,54, 54))
    np.random.shuffle(subs_list)
    test_size = int(test_rate* len(subs_list))
    test_subs, train_subs = subs_list[:test_size],subs_list[test_size:]
    print(test_subs)
    source_test_x = []
    source_test_y = []
    for sub in test_subs:
        target_session_1_path = os.path.join(data_path,r'sub{}_sess1_train/Data.mat'.format(sub))
        target_session_2_path = os.path.join(data_path,r'sub{}_sess2_train/Data.mat'.format(sub))

        session_1_data = sio.loadmat(target_session_1_path)
        session_2_data = sio.loadmat(target_session_2_path)
        R = None
        if is_few_EA is True:
            session_1_x = EA(session_1_data['x_data'],R)
        else:
            session_1_x = session_1_data['x_data']
            
        if is_few_EA is True:
            session_2_x = EA(session_2_data['x_data'],R)
        else:
            session_2_x = session_2_data['x_data']
        
        # -- debug for BCIC 2b
        test_x_1 = torch.FloatTensor(session_1_x)      
        test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

        test_x_2 = torch.FloatTensor(session_2_x)      
        test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
        
        if target_sample>0:
            test_x_1 = temporal_interpolation(test_x_1, target_sample)
            test_x_2 = temporal_interpolation(test_x_2, target_sample)
        source_test_x.extend([test_x_1, test_x_2])
        source_test_y.extend([test_y_1, test_y_2])
            
    test_dataset = eeg_dataset(torch.cat(source_test_x,dim=0),torch.cat(source_test_y,dim=0))

    source_train_x = []
    source_train_y = []
    for i in train_subs:
        train_path = os.path.join(data_path,r'sub{}_sess1_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_sess2_train/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)
        
        source_train_x.append(session_1_x)
        source_train_y.append(session_1_y)


        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'],R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)
        
        source_train_x.append(session_2_x)
        source_train_y.append(session_2_y)
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))

    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y)
    
    return train_dataset,test_dataset

def get_data_Nakanishi2015(sub,data_path="Data/Nakanishi2015_8_64HZ/",few_shot_number = 1, is_few_EA = False, target_sample=-1):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample)
        
    test_dataset = eeg_dataset(test_x_1,test_y_1)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in tqdm(range(1,11)):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        # print(train_path)
        train_data = sio.loadmat(train_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 40,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset

def get_data_Wang2016(sub,data_path="Data/Wang2016_4_20HZ/",few_shot_number = 1, is_few_EA = False, target_sample=-1):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample)
        
    test_dataset = eeg_dataset(test_x_1,test_y_1)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in tqdm(range(1,36)):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        # print(train_path)
        train_data = sio.loadmat(train_path)
        
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 40,stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample)
        
    train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset
if __name__=="__main__":
    train_dataset,valid_dataset,test_dataset = get_data_Wang2016(1,"Data/Wang2016_4_20HZ/", 1, is_few_EA = False, target_sample=-1)
    # # train_dataset,valid_dataset,test_dataset = get_data(1,data_path,1,True)
    # avg_ent = 0 

    # for i in range(1000):
    #     # print(geban()) 
    #     # print(sample())
    #     num_class = geban_entropy(entropy_scope=[1.2,1e6])#geban()#sample()
    #     total = sum(num_class)
    #     num_class = [x/total for x in num_class]
    #     # print(num_class)
    #     # print(sum([-x*(math.log(x)) for x in num_class if x>0]))
    #     ent = stats.entropy(num_class) 
    #     avg_ent+=ent
    #     print(avg_ent/1000) # sample 1.2110981470145854 geban 0.9734407215366253
    
    
import mne
import torch
import tqdm
import pandas as pd 
import csv
import numpy as np
import os
import scipy.io as scio

import random
mne.set_log_level("ERROR")

def min_max_normalize(x: torch.Tensor, data_max=None, data_min=None, low=-1, high=1):
    if data_max is not None:
        max_scale = data_max - data_min
        scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
        
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    x  = (high - low) * x
    
    if data_max is not None:
        x = torch.cat([x, torch.ones((1, x.shape[-1])).to(x)*scale])
    return x
    