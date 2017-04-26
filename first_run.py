import os
import train
import sys

def compute_file_name(pcov, pfc):
    name = ''
    name += 'cov' + str(int(pcov[0] * 10))
    name += 'cov' + str(int(pcov[1] * 10))
    name += 'fc' + str(int(pfc[0] * 10))
    name += 'fc' + str(int(pfc[1] * 10))
    name += 'fc' + str(int(pfc[2] * 10))
    return name

acc_list = []
count = 0
# pcov = [10., 66.]
# pfc = [85., 66., 10.]
pcov = [0., 0.]
pfc = [0., 0., 0.]
retrain = 0
f_name = compute_file_name(pcov, pfc)
parent_dir = './'
# lr = 1e-5
lr = 1e-3
with_biases = False
param = [
    ('-pcov1',pcov[0]),
    ('-pcov2',pcov[1]),
    ('-pfc1',pfc[0]),
    ('-pfc2',pfc[1]),
    ('-pfc3',pfc[2]),
    ('-first_time', False),
    ('-file_name', f_name),
    ('-train', True),
    ('-prune', False),
    ('-lr', lr),
    ('-with_biases', with_biases),
    ('-parent_dir', parent_dir),
    ('-lambda1', 1e-4),
    ('-lambda2', 1e-5)
    ]

_ = train.main(param)
