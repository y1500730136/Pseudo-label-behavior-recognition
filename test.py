import logging

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# fake_data = torch.load('./fake/hmdb51_fake_data.pth')
# fake_label = torch.full((fake_data.shape[0], ), -1)
# fake_dataset = TensorDataset(fake_data, fake_label)
# fake_dataloader = DataLoader(dataset=fake_dataset, batch_size=32, shuffle=False, num_workers=1)
# true_dataloader = DataLoader(dataset=fake_dataset, batch_size=32, shuffle=False, num_workers=1)
#
# for i, ((true_data, true_target), (fake_data, fake_label)) in enumerate(zip(true_dataloader, fake_dataloader)):
#     print(true_data.shape)
#     print(fake_data.shape)

test1 = torch.full((8, 3), 0)
test2 = torch.full((8, 3), 1)

test3 = test1.data.clone()

print(test1)

label = torch.Tensor([-1, -1, -1, 0, 0, 0, 0, -1])
uncond = (label < 0)
unnnz = torch.nonzero(uncond)
test3.data[unnnz] = test2.data[unnnz]

print(test1)
