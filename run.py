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
lr = 1e-4
with_biases = False

# initial run
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
    ('-lr', lr),
    ('-with_biases', with_biases),
    ('-parent_dir', parent_dir),
    ('-lambda1', 1e-4),
    ('-lambda2', 1e-5)
    ]
test_acc = train.main(param)
print("first train")
acc_list.append((pcov[:],pfc[:],test_acc))
print('accuracy summary: {}'.format(acc_list))

run = 1
hist = [(pcov, pfc, test_acc)]
pcov = [0., 0.]
pfc = [0., 0., 0.]
retrain_cnt = 0
roundrobin = 0
with_biases = True
# Prune
while (run):
    param = [
        ('-pcov1',pcov[0]),
        ('-pcov2',pcov[1]),
        ('-pfc1',pfc[0]),
        ('-pfc2',pfc[1]),
        ('-pfc3',pfc[2]),
        ('-first_time', False),
        ('-file_name', f_name),
        ('-train', False),
        ('-prune', True),
        ('-lr', lr),
        ('-with_biases', with_biases),
        ('-parent_dir', parent_dir),
        ('-lambda1', 1e-4),
        ('-lambda2', 1e-5)
        ]
    _ = train.main(param)

    # pruning saves the new models, masks
    f_name = compute_file_name(pcov, pfc)

    # TRAIN
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
        ('-lr', lr),
        ('-with_biases', with_biases),
        ('-parent_dir', parent_dir),
        ('-lambda1', 1e-4),
        ('-lambda2', 1e-5)
        ]
    acc = train.main(param)
    hist.append((pcov, pfc, acc))
    f_name = compute_file_name(pcov, pfc)
    # pcov[1] = pcov[1] + 10.
    if (acc > 0.808):
        pfc[0] = pfc[0] + 10.
        lr = 1e-4
        retrain = 0
        acc_list.append((pcov,pfc,acc))
    else:
        retrain = retrain + 1
        if (retrain == 5):
            lr = lr / float(2)
        if (retrain == 10):
            lr = lr / float(5)
        if (retrain > 15):
            break
    count = count + 1
    print('accuracy summary: {}'.format(acc_list))
    print (acc)

print('accuracy summary: {}'.format(acc_list))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
with open("acc_cifar.txt", "w") as f:
    for item in acc_list:
        f.write("{} {} {}\n".format(item[0],item[1],item[2]))
