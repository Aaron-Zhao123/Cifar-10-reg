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
crates = {
    'cov1': 0.,
    'cov2': 0.,
    'fc1': 1.75,
    'fc2': 0.,
    'fc3': 0.
}
retrain = 0

parent_dir = '/Users/aaron/Projects/Mphil_project/tmp_cifar10/async_pruning/'
parent_dir = './assets/'
with_biases = False

run = 1
hist = []
lr = 1e-5
# # pcov = [0., 40.]
# pcov = [0., 0.]
# # pfc = [90., 20., 0.]
# pfc = [0., 0., 0.]
# Prune
while (run):

    # TEST
    param = [
        ('-cRates', crates),
        ('-first_time', False),
        ('-train', False),
        ('-prune', False),
        ('-lr', lr),
        ('-with_biases', with_biases),
        ('-parent_dir', parent_dir),
        ('-lambda1', 1e-4),
        ('-lambda2', 1e-5)
        ]
    acc = train.main(param)
    print(acc)
    run = 0
    acc_list.append(acc)
    count = count + 1
    print (acc)

print('accuracy summary: {}'.format(hist))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
with open("acc_cifar.txt", "w") as f:
    for item in acc_list:
        f.write("%s\n"%item)
