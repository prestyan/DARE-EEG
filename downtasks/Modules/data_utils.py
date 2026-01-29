import os
import random
import scipy
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split


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


def EA(x,new_R = None):
    # print(x.shape)
    '''
    The Eulidean space alignment approach for EEG data.

    Arg:
        x:The input data,shape of NxCxS
        new_R：The reference matrix.
    Return:
        The aligned data.
    '''
    
    xt = np.transpose(x,axes=(0,2,1))
    # print('xt shape:',xt.shape)
    E = np.matmul(x,xt)
    # print(E.shape)
    R = np.mean(E, axis=0)
    # print('R shape:',R.shape)

    R_mat = scipy.linalg.fractional_matrix_power(R,-0.5)
    new_x = np.einsum('n c s,r c -> n r s',x,R_mat)
    if new_R is None:
        return new_x

    new_x = np.einsum('n c s,r c -> n r s',new_x,scipy.linalg.fractional_matrix_power(new_R,0.5))
    
    return new_x

def StableEA(x, new_R=None, eps=1e-6, normalize='time'):
    """
    x: (N, C, S)
    new_R: (C, C) global reference covariance (R_ref). If None -> whiten to I.
    """
    # (N, S, C)
    xt = np.transpose(x, axes=(0, 2, 1))

    # (N, C, C) = X X^T
    E = np.matmul(x, xt)

    if normalize == 'time':
        # divide by time length S
        E = E / x.shape[-1]
    elif normalize == 'trace':
        tr = np.trace(E, axis1=1, axis2=2)  # (N,)
        E = E / (tr[:, None, None] + eps)

    R = np.mean(E, axis=0)  # (C, C)
    R = (R + R.T) / 2.0
    R = R + eps * np.eye(R.shape[0], dtype=R.dtype)

    R_inv_sqrt = scipy.linalg.fractional_matrix_power(R, -0.5)
    # whiten: (C,C) @ (N,C,S) on channel dim
    xw = np.einsum('rc,ncs->nrs', R_inv_sqrt, x)

    if new_R is None:
        return xw

    new_R = (new_R + new_R.T) / 2.0
    new_R = new_R + eps * np.eye(new_R.shape[0], dtype=new_R.dtype)
    R_ref_sqrt = scipy.linalg.fractional_matrix_power(new_R, 0.5)

    # color to R_ref
    xa = np.einsum('rc,ncs->nrs', R_ref_sqrt, xw)
    return xa


def compute_R_ref(data_path, target_sub, eps=1e-6):
    Rs = []
    for i in range(1, 10):
        if i == target_sub:
            continue

        train_path = os.path.join(data_path, f"sub{i}_train/Data.mat")
        train_data = sio.loadmat(train_path)
        x = train_data['x_data']  # (N,C,S)

        xt = np.transpose(x, (0,2,1))
        E = np.matmul(x, xt) / x.shape[-1]  # /S
        R = np.mean(E, axis=0)
        R = (R + R.T) / 2.0
        R = R + eps * np.eye(R.shape[0], dtype=R.dtype)
        Rs.append(R)

    R_ref = np.mean(np.stack(Rs, axis=0), axis=0)
    R_ref = (R_ref + R_ref.T) / 2.0
    R_ref = R_ref + eps * np.eye(R_ref.shape[0], dtype=R_ref.dtype)
    return R_ref


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

def get_data(sub,data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1, use_avg=True):
    # 1) 先算全局参考（只用 source train）
    R_ref = None
    if is_few_EA:
        R_ref = compute_R_ref(data_path, target_sub=sub)
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None
    if is_few_EA is True:
        session_1_x = StableEA(session_1_data['x_data'], new_R=R_ref)
    else:
        session_1_x = session_1_data['x_data']
        
    if is_few_EA is True:
        session_2_x = StableEA(session_2_data['x_data'], new_R=R_ref)
    else:
        session_2_x = session_2_data['x_data']
    
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample, use_avg=use_avg)
        test_x_2 = temporal_interpolation(test_x_2, target_sample, use_avg=use_avg)

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
            session_1_x = StableEA(train_data['x_data'], new_R=R_ref)
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
            session_2_x = StableEA(test_data['x_data'], new_R=R_ref)
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
        
    train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset