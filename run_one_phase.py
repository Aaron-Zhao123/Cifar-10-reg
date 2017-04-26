import os
import train
import sys
# os.system('python training_v3.py -p0')
# os.system('python training_v3.py -p1')
# os.system('python training_v3.py -p2')
# os.system('python training_v3.py -p3')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p5')
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
pcov = [0., 40.]
pfc = [85., 40., 0.]
retrain = 0
f_name = compute_file_name(pcov, pfc)

# f_name = 'pruningv00'
# initial run
param = [
    ('-pcov1',pcov[0]),
    ('-pcov2',pcov[1]),
    ('-pfc1',pfc[0]),
    ('-pfc2',pfc[1]),
    ('-pfc3',pfc[2]),
    ('-first_time', False),
    ('-file_name', f_name),
    ('-train', True),
    ('-prune', False)
    ]
# acc = train.main(param)
param = [
    ('-pcov1',pcov[0]),
    ('-pcov2',pcov[1]),
    ('-pfc1',pfc[0]),
    ('-pfc2',pfc[1]),
    ('-pfc3',pfc[2]),
    ('-first_time', False),
    ('-file_name', f_name),
    ('-train', False),
    ('-prune', False)
    ]
test_acc = train.main(param)
print("first train")
acc_list.append((pcov,pfc,test_acc))
print('accuracy summary: {}'.format(acc_list))

run = 1

level1 = 1
level2 = 0
level3 = 0
level4 = 0
level5 = 0
level6 = 0

working_level = level1
hist = [(pcov, pfc, test_acc)]
pcov = [0., 50.]
pfc = [85., 50., 0.]
retrain_cnt = 0
roundrobin = 0
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
        ('-prune', True)
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
        ('-prune', False)
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
        ('-prune', False)
        ]
    acc = train.main(param)
    hist.append((pcov, pfc, acc))
    f_name = compute_file_name(pcov, pfc)
    # pcov[1] = pcov[1] + 10.
    if (acc > 0.823):

        # pfc[1] = pfc[1] + 10.
        pcov[1] = pcov[1] + 10.
        retrain = 0
        acc_list.append((pcov,pfc,acc))
    else:
        retrain = retrain + 1
        if (retrain > 5):
            break
    # pcov[1] = pcov[1] + 10.
    # if (pfc[0] > 90.):
    #     run = 0
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

    count = count + 1
    print('accuracy summary: {}'.format(acc_list))
    print (acc)

print('accuracy summary: {}'.format(acc_list))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
with open("acc_cifar.txt", "w") as f:
    for item in acc_list:
        f.write("{} {} {}\n".format(item[0],item[1],item[2]))
