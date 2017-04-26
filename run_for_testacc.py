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
pcov = [0., 0.]
pfc = [0., 0., 0.]
retrain = 0
f_name = compute_file_name(pcov, pfc)

# f_name = 'pruningv00'
# initial run
parent_dir = '/Users/aaron/Projects/Mphil_project/tmp_cifar10/async_pruning/'
parent_dir = 'assets/withbiases/'
parent_dir = './'

run = 1
hist = []
lr = 1e-5
# # pcov = [0., 40.]
# pcov = [0., 0.]
# # pfc = [90., 20., 0.]
# pfc = [0., 0., 0.]
# Prune
while (run):

    f_name = compute_file_name(pcov, pfc)

    # TEST
    param = [
        ('-pcov1',pcov[0]),
        ('-pcov2',pcov[1]),
        ('-pfc1',pfc[0]),
        ('-pfc2',pfc[1]),
        ('-pfc3',pfc[2]),
        ('-first_time', False),
        ('-file_name', f_name),
        ('-train', False),
        ('-prune', False),
        ('-parent_dir', parent_dir),
        ('-lr', lr),
        ('-parent_dir', parent_dir),
        ('-lambda1', 1e-3),
        ('-lambda2', 1e-3)
        ]
    acc = train.main(param)
    hist.append((pcov[:], pfc[:], acc))
    print(pfc)
    print(hist)
    f_name = compute_file_name(pcov, pfc)
    # pcov[1] = pcov[1] + 10.
    # pfc[0] = pfc[0] + 10.
    # if (pfc[0] > 90):
        # run = 0
    run = 0
    # if (working_level == level1):
    #     if (acc >= 0.8):
    #         f_name = compute_file_name(pcov, pfc)
    #         pfc[0] = pfc[0] + 10.
    #     else:
    #         pfc[0] = pfc[0] - 10.
    #         f_name = compute_file_name(pcov, pfc)
    #         working_level = level2
    #         run = 0
    # if (working_level == level2):
    #     if (acc >= 0.8):
    #         f_name = compute_file_name(pcov, pfc)
    #         pcov = []pcov + [1., 1.]
    #         pfc =  pfc +[1., 1., 1.]
    #     else:
    #         pcov = pcov - [1., 1.]
    #         pfc =  pfc -[1., 1., 1.]
    #         f_name = compute_file_name(pcov, pfc)
    #         pcov = pcov + [0.1, 0.1]
    #         pfc =  pfc +[0.1, 0.1, 0.1]
    #         working_level = level3
    # if (working_level == level3):
    #     if (acc >= 0.8):
    #         f_name = compute_file_name(pcov, pfc)
    #         pcov = pcov + [0.1, 0.1]
    #         pfc =  pfc +[0.1, 0.1, 0.1]
    #     else:
    #         run = 0
    #         print('finished')
    #
    #

    acc_list.append(acc)
    count = count + 1
    print (acc)

print('accuracy summary: {}'.format(hist))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
with open("acc_cifar.txt", "w") as f:
    for item in acc_list:
        f.write("%s\n"%item)
