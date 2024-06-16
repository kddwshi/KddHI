import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import itertools
from torch.distributions.categorical import Categorical


class MaskLayer1d(nn.Module):

    def __init__(self, value, append):
        super().__init__()
        self.value = value
        self.append = append

    def forward(self, input_tuple):
        x, S = input_tuple
        x = x * S + self.value * (1 - S) #VALUE A 0
        if self.append:
            x = torch.cat((x, S), dim=1)
        return x


class MaskLayer2d(nn.Module):

    def __init__(self, value, append):
        super().__init__()
        self.value = value
        self.append = append

    def forward(self, input_tuple):
        x, S = input_tuple
        if len(S.shape) == 3:
            S = S.unsqueeze(1)
        x = x * S + self.value * (1 - S)
        if self.append:
            x = torch.cat((x, S), dim=1)
        return x


class KLDivLoss(nn.Module):
    
    def __init__(self, reduction='batchmean', log_target=False):
        super().__init__()
        self.kld = nn.KLDivLoss(reduction=reduction, log_target=log_target)

    def forward(self, pred, target):
        return self.kld(pred.log_softmax(dim=1), target)
    

class DatasetRepeat(Dataset):
    '''
    A wrapper around multiple datasets that allows repeated elements when the
    dataset sizes don't match. The number of elements is the maximum dataset
    size, and all datasets must be broadcastable to the same size.
    Args:
      datasets: list of dataset objects.
    '''

    def __init__(self, datasets):
        # Get maximum number of elements.
        assert np.all([isinstance(dset, Dataset) for dset in datasets])
        items = [len(dset) for dset in datasets]
        num_items = np.max(items)

        # Ensure all datasets align.
        # assert np.all([num_items % num == 0 for num in items])
        self.dsets = datasets
        self.num_items = num_items
        self.items = items

    def __getitem__(self, index):
        assert 0 <= index < self.num_items
        return_items = [dset[index % num] for dset, num in
                        zip(self.dsets, self.items)]
        return tuple(itertools.chain(*return_items))

    def __len__(self):
        return self.num_items


class DatasetInputOnly(Dataset):
    '''
    A wrapper around a dataset object to ensure that only the first element is
    returned.
    Args:
      dataset: dataset object.
    '''

    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset

    def __getitem__(self, index):
        return (self.dataset[index][0],)

    def __len__(self):
        return len(self.dataset)


class UniformSampler:

    def __init__(self, num_players):
        self.num_players = num_players

    def sample(self, batch_size, paired_sampling=False):
        '''
        Generate sample.
        Args:
          batch_size:
        '''
        S = torch.ones(batch_size, self.num_players, dtype=torch.float32)
        num_included = (torch.rand(batch_size) * (self.num_players + 1)).int()
        # TODO ideally avoid for loops
        # TODO ideally pass buffer to assign samples in place
        for i in range(batch_size):
            if paired_sampling and i % 2 == 1:
                S[i] = 1 - S[i - 1]
            else:
                S[i, num_included[i]:] = 0
                S[i] = S[i, torch.randperm(self.num_players)]

        return S


class ShapleySampler:

    def __init__(self, num_players):
        arange = torch.arange(1, num_players)
        w = 1 / (arange * (num_players - arange))
        w = w / torch.sum(w)
        self.categorical = Categorical(probs=w)
        self.num_players = num_players
        self.tril = torch.tril(
            torch.ones(num_players - 1, num_players, dtype=torch.float32),
            diagonal=0)

    def sample(self, batch_size, paired_sampling):
        '''
        Generate sample.
        Args:
          batch_size: number of samples.
          paired_sampling: whether to use paired sampling.
        '''
        num_included = 1 + self.categorical.sample([batch_size])
        S = self.tril[num_included - 1]
        # TODO ideally avoid for loops
        for i in range(batch_size):
            if paired_sampling and i % 2 == 1:
                S[i] = 1 - S[i - 1]
            else:
                S[i] = S[i, torch.randperm(self.num_players)]
        return S

class MultiTaskModel(nn.Module):
    def __init__(self, Layer_size, num_features):
        super(MultiTaskModel,self).__init__()
        self.body = nn.Sequential(
            MaskLayer1d(value=0, append=True),
            nn.Linear(2 * num_features, Layer_size), #FA IL CAT DELLA MASK (SUBSET S)
            nn.ReLU(inplace=True),
            nn.Linear(Layer_size, Layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(Layer_size, Layer_size),
            nn.ReLU(inplace=True),
        )
        self.head1 = nn.Sequential(
            nn.Linear(Layer_size, 2),
        )
        self.head2 = nn.Sequential(
            nn.Linear(Layer_size, 2),
        )

    def forward(self,x):
        x = self.body(x)
        v1 = self.head1(x)
        v2 = self.head2(x)
        return v1, v2

def get_grand_null_u(x_t, N, y, surrogate_VV):
    link=nn.Softmax(dim=-1)
    # x_t=torch.tensor(x, dtype=torch.float32)
    ones = torch.ones(1, N, dtype=torch.float32)
    v1, v2 = surrogate_VV(x_t, ones)
    v1 = link(v1[0])
    v2 = link(v2[0])
    v1 = v1.data.numpy()
    v2 = v2.data.numpy()
    if y==0:
        v_u = np.array([-(v2[1]-v1[0])/(2), (v2[1]-v1[0])/(2)])
    else:
        v_u = np.array([-(v1[1]-v2[0])/(2), (v1[1]-v2[0])/(2)])
        
    zeros = torch.ones(1, N, dtype=torch.float32)
    v1, v2 = surrogate_VV(x_t, zeros)
    v1 = link(v1[0])
    v2 = link(v2[0])
    v1 = v1.data.numpy()
    v2 = v2.data.numpy()
    if y==0:
        v_u_null = np.array([-(v2[1]-v1[0])/(2), (v2[1]-v1[0])/(2)])
    else:
        v_u_null = np.array([-(v1[1]-v2[0])/(2), (v1[1]-v2[0])/(2)])
    diff_u = np.array([v_u[0] - v_u_null[1], v_u[1] - v_u_null[0]])
    return v_u, v_u_null, diff_u


def ref_phi(median_phi, x, y, N, diff_u):
    interval=[]
    for el in median_phi[:,y]:
        tmp = np.array([el,el])
        tmp = tmp + diff_u/N # NORMALIZED
        interval.append(tmp)
    interval=np.array(interval)
    return interval

def ref_phi_U(median_phi, x, y, N, diff_u):
    interval=[]
    for idx,el in enumerate(median_phi[:,y]):
        tmp = np.array([el,el])
        normaliz=N[idx]/np.sum(N)
        tmp = tmp + diff_u*normaliz # NORMALIZED
        interval.append(tmp)
    interval=np.array(interval)
    return interval

class MedianModel():
    def __init__(self, imputer, link, device, N):
        self.imputer = imputer
        self.link = link
        self.device = device
        self.N = N

    def predict(self, input):
        out=[]
        for x in input:
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            ones = torch.ones(x_t.shape[0], self.N, dtype=torch.float32)
            # print(x_t.shape, ones.shape)
            v1, v2 = self.imputer(x_t, ones)
            # print(v1.shape, v2.shape)
            v1 = self.link(v1[0])
            v2 = self.link(v2[0])
            v1 = v1.data.numpy()
            v2 = v2.data.numpy()
            pred=np.array([(v1[0]+v2[1])/2, (v2[0]+v1[1])/2])
            # print(pred.shape)
            out.append(pred)
        return np.array(out)
    
    def predict_proba(self, input):
        return self.predict(input)

class UncertainModel():
    def __init__(self, imputer, link, device, N):
        self.imputer = imputer
        self.link = link
        self.device = device
        self.N = N

    def predict(self, input):
        out=[]
        for x in input:
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            ones = torch.ones(x_t.shape[0], self.N, dtype=torch.float32)
            # print(x_t.shape, ones.shape)
            v1, v2 = self.imputer(x_t, ones)
            # print(v1.shape, v2.shape)
            v1 = self.link(v1[0])
            v2 = self.link(v2[0])
            v1 = v1.data.numpy()
            v2 = v2.data.numpy()
            # print(v1, v2)
            pred=np.array([np.abs(v1[0]-v2[1])/2, np.abs(v2[0]-v1[1])/2])
            # print(pred.shape)
            out.append(pred)
        return np.array(out)
    
    def predict_proba(self, input):
        return self.predict(input)