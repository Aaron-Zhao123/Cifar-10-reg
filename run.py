import os
import train
import sys

def compute_file_name(p):
    name = ''
    name += 'cov' + str(int(round(p['cov1'] * 10)))
    name += 'cov' + str(int(round(p['cov2'] * 10)))
    name += 'fc' + str(int(round(p['fc1'] * 100)))
    name += 'fc' + str(int(round(p['fc2'] * 10)))
    name += 'fc' + str(int(round(p['fc3'] * 10)))
    return name

acc_list = []
count = 0
retrain = 0
parent_dir = 'assets/'
# lr = 1e-5
lr = 1e-4
crates = {
    'cov1': 0.4,
    'cov2': 1.6,
    'fc1': 2.04,
    'fc2': 1.4,
    'fc3': 0.9
}
retrain_cnt = 0
roundrobin = 0
with_biases = False
lambda1 = 1e-5
lambda2 = 1e-5
# Prune
while (crates['fc2'] < 1.8):
    count = 0
    iter_cnt = 0
    while (iter_cnt < 7):
        if (iter_cnt > 3 and iter_cnt < 5):
            lr = 5e-5
        elif (iter_cnt >= 5):
            lr = 1e-5
        else:
            lr = 1e-4
        param = [
            ('-cRates', crates),
            ('-first_time', False),
            ('-train', False),
            ('-prune', True),
            ('-lr', lr),
            ('-with_biases', with_biases),
            ('-parent_dir', parent_dir),
            ('-lambda1', lambda1),
            ('-lambda2', lambda2)
            # ('-lambda1', 0.),
            # ('-lambda2', 0.)
            ]
        # _ = train.main(param)

        # TRAIN
        param = [
            ('-cRates', crates),
            ('-first_time', False),
            ('-train', True),
            ('-prune', False),
            ('-lr', lr),
            ('-with_biases', with_biases),
            ('-parent_dir', parent_dir),
            ('-lambda1', lambda1),
            ('-lambda2', lambda2)
            ]
        _ = train.main(param)

        # TEST
        param = [
            ('-cRates', crates),
            ('-first_time', False),
            ('-train', False),
            ('-prune', False),
            ('-lr', lr),
            ('-with_biases', with_biases),
            ('-parent_dir', parent_dir),
            ('-lambda1', lambda1),
            ('-lambda2', lambda2)
            ]
        acc = train.main(param)

        if (acc > 0.82 or iter_cnt == 7):
            file_name = compute_file_name(crates)
            # crates['fc1'] = crates['fc1'] + 0.05
            # crates['cov2'] = crates['cov2'] + 0.2
            # crates['fc2'] = crates['fc2'] + 0.1
            # crates['fc3'] = crates['fc3'] + 0.2
            # crates['cov2'] = crates['cov2'] + 0.5
            # crates['cov1'] = crates['cov1'] + 0.2
            acc_list.append((crates,acc))
            param = [
                ('-first_time', False),
                ('-train', False),
                ('-prune', False),
                ('-lr', lr),
                ('-with_biases', with_biases),
                ('-parent_dir', parent_dir),
                ('-iter_cnt',iter_cnt),
                ('-cRates',crates),
                ('-save', True),
                ('-lambda1', lambda1),
                ('-lambda2', lambda2),
                ('-org_file_name', file_name)
                ]
            # _ = train.main(param)
            break
        else:
            iter_cnt = iter_cnt + 1
    print('accuracy summary: {}'.format(acc_list))


print('accuracy summary: {}'.format(acc_list))
# acc_list = [0.82349998, 0.8233, 0.82319999, 0.81870002, 0.82050002, 0.80400002, 0.74940002, 0.66060001, 0.5011]
with open("acc_cifar.txt", "w") as f:
    for item in acc_list:
        f.write("{} {} {}\n".format(item[0],item[1],item[2]))
