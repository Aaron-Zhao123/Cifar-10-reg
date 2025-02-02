import cifar10
import tensorflow as tf
import numpy as np
import sys
import os
import pickle
import time
import getopt

from cifar10 import img_size, num_channels, num_classes

class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    Returns:
    Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def initialize_variables(exist, file_name):
    NUM_CHANNELS = 3
    IMAGE_SIZE = 32
    NUM_CLASSES = 10
    keys = ['cov1','cov2','fc1','fc2','fc3']
    if (exist == 1):
        with open(file_name, 'rb') as f:
            (weights_val, biases_val) = pickle.load(f)
        weights = {
            'cov1': tf.Variable(weights_val['cov1']),
            'cov2': tf.Variable(weights_val['cov2']),
            'fc1': tf.Variable(weights_val['fc1']),
            'fc2': tf.Variable(weights_val['fc2']),
            'fc3': tf.Variable(weights_val['fc3'])
        }
        biases = {
            'cov1': tf.Variable(biases_val['cov1']),
            'cov2': tf.Variable(biases_val['cov2']),
            'fc1': tf.Variable(biases_val['fc1']),
            'fc2': tf.Variable(biases_val['fc2']),
            'fc3': tf.Variable(biases_val['fc3'])
        }
    else:
        weights = {
            'cov1': tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 64],
                                                        stddev=5e-2)),
            'cov2': tf.Variable(tf.truncated_normal([5, 5, 64, 64],
                                                        stddev=5e-2)),
            'fc1': tf.Variable(tf.truncated_normal([6 * 6 * 64, 384],
                                                        stddev=0.04)),
            'fc2': tf.Variable(tf.random_normal([384, 192],
                                                        stddev=0.04)),
            'fc3': tf.Variable(tf.random_normal([192, NUM_CLASSES],
                                                        stddev=1/192.0))
        }
        biases = {
            'cov1': tf.Variable(tf.constant(0.1, shape=[64])),
            'cov2': tf.Variable(tf.constant(0.1, shape=[64])),
            'fc1': tf.Variable(tf.constant(0.1, shape=[384])),
            'fc2': tf.Variable(tf.constant(0.1, shape=[192])),
            'fc3': tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES]))
        }
    return (weights, biases)

def prune_weights(cRates, weights, weights_mask, biases, biases_mask, mask_dir, f_name):
    keys = ['cov1','cov2','fc1','fc2','fc3']
    new_mask = {}
    for key in keys:
        w_eval = weights[key].eval()
        threshold_off = 0.9*(np.mean(w_eval) + cRates[key] * np.std(w_eval))
        threshold_on = 1.1*(np.mean(w_eval) + cRates[key] * np.std(w_eval))
        mask_off = np.abs(w_eval) < threshold_off
        mask_on = np.abs(w_eval) > threshold_on
        new_mask[key] = np.logical_or(((1 - mask_off) * weights_mask[key]),mask_on).astype(int)
    with open(mask_dir + f_name, 'wb') as f:
        pickle.dump((new_mask,biases_mask), f)

def initialize_weights_mask(first_time_training, mask_dir, file_name):
    NUM_CHANNELS = 3
    NUM_CLASSES = 10
    if (first_time_training == 1):
        print('setting initial mask value')
        weights_mask = {
            'cov1': np.ones([5, 5, NUM_CHANNELS, 64]),
            'cov2': np.ones([5, 5, 64, 64]),
            'fc1': np.ones([6 * 6 * 64, 384]),
            'fc2': np.ones([384, 192]),
            'fc3': np.ones([192, NUM_CLASSES])
        }
        biases_mask = {
            'cov1': np.ones([64]),
            'cov2': np.ones([64]),
            'fc1': np.ones([384]),
            'fc2': np.ones([192]),
            'fc3': np.ones([NUM_CLASSES])
        }

        with open(mask_dir + 'maskcov0cov0fc0fc0fc0.pkl', 'wb') as f:
            pickle.dump((weights_mask,biases_mask), f)
    else:
        with open(mask_dir + file_name,'rb') as f:
            (weights_mask, biases_mask) = pickle.load(f)
    return (weights_mask, biases_mask)

def prune_info(weights, counting):
    if (counting == 0):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'].eval())
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
        # print(weights['cov1'].eval())
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'].eval())
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'].eval())
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc3'].eval())
        print('fc3 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
    if (counting == 1):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'])
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
        # print(weights['cov1'].eval())
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'])
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'])
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'])
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc3'])
        print('fc3 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
def plot_weights(weights,pruning_info):
        keys = ['cov1','cov2','fc1', 'fc2','fc2']
        fig, axrr = plt.subplots( 2, 2)  # create figure &  axis
        fig_pos = [(0,0), (0,1), (1,0), (1,1)]
        index = 0
        for key in keys:
            weight = weights[key].eval().flatten()
            # print (weight)
            size_weight = len(weight)
            weight = weight.reshape(-1,size_weight)[:,0:size_weight]
            x_pos, y_pos = fig_pos[index]
            #take out zeros
            weight = weight[weight != 0]
            # print (weight)
            hist,bins = np.histogram(weight, bins=100)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            axrr[x_pos, y_pos].bar(center, hist, align = 'center', width = width)
            axrr[x_pos, y_pos].set_title(key)
            index = index + 1
        fig.savefig('fig_v3/weights'+pruning_info)
        plt.close(fig)


def cov_network(images, weights, biases, keep_prob):
    BATCH_SIZE = 128
    NUM_CLASSES = 10
    IMAGE_SIZE = 32
    NUM_CHANNELS = 3
    # cov1_weight = weights['cov1'] * weights_mask['cov1']
    # conv1
    conv = tf.nn.conv2d(images, weights['cov1'], [1, 1, 1, 1], padding='SAME')
    # conv = tf.nn.conv2d(images, cov1_weight, [1, 1, 1, 1], padding='SAME')

    pre_activation = tf.nn.bias_add(conv, biases['cov1'])
    conv1 = tf.nn.relu(pre_activation)
    l1_cov1 = tf.reduce_sum(tf.abs(weights['cov1']))
    l2_cov1 = tf.nn.l2_loss(weights['cov1'])

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    # conv2
    # cov2_weight = weights['cov2'] * weights_mask['cov2']
    conv = tf.nn.conv2d(norm1, weights['cov2'], [1, 1, 1, 1], padding='SAME')
    # conv = tf.nn.conv2d(norm1, cov2_weight, [1, 1, 1, 1], padding='SAME')
    l1_cov2 = tf.reduce_sum(tf.abs(weights['cov2']))
    l2_cov2 = tf.nn.l2_loss(weights['cov2'])

    pre_activation = tf.nn.bias_add(conv, biases['cov2'])
    conv2 = tf.nn.relu(pre_activation)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    # Move everything into depth so we can perform a single matrix multiply.
    # dim = 1
    # for d in pool2.get_shape()[1:].as_list():
    #   dim *= d
    # print(pool2.get_shape().as_list())
    # reshape = tf.reshape(pool2, [-1, IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64])
    reshape = tf.reshape(pool2, [-1, 6 * 6 * 64])
    # reshape = tf.reshape(pool2, [BATCH_SIZE, dim])
    # print(reshape)

    # fc1_weight = weights['fc1'] * weights_mask['fc1']
    local3 = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    # dropout
    local3_drop = tf.nn.dropout(local3, keep_prob)

    l1_fc1 = tf.reduce_sum(tf.abs(weights['fc1']))
    l2_fc1 = tf.nn.l2_loss(weights['fc1'])
# # local4
    # fc2_weight = weights['fc2'] * weights_mask['fc2']
    local4 = tf.nn.relu(tf.matmul(local3_drop, weights['fc2']) + biases['fc2'])
    local4_drop = tf.nn.dropout(local4, keep_prob)

    l1_fc2 = tf.reduce_sum(tf.abs(weights['fc2']))
    l2_fc2 = tf.nn.l2_loss(weights['fc2'])
# # We don't apply softmax here because
# # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
# # and performs the softmax internally for efficiency.
    # fc3_weight = weights['fc3'] * weights_mask['fc3']
    softmax_linear = tf.add(tf.matmul(local4_drop, weights['fc3']), biases['fc3'])

    l1_fc3 = tf.reduce_sum(tf.abs(weights['fc3']))
    l2_fc3 = tf.nn.l2_loss(weights['fc3'])
    l1 = l1_cov1 + l1_cov2 + l1_fc1 + l1_fc2 + l1_fc3
    l2 = l2_cov1 + l2_cov2 + l2_fc1 + l2_fc2 + l2_fc3
    return (l1, l2, softmax_linear)

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

class training_data():
    def __init__ (self, images, labels):
        self.images = images
        self.labels = labels
        self.batch_cnt = 0
        self.data_size = len(images)
    def feed_next_batch(self, batch_size):
        start_pt = self.batch_cnt
        self.batch_cnt += batch_size
        if (self.batch_cnt <= self.data_size):
            return (self.images[start_pt:self.batch_cnt],
                    self.labels[start_pt:self.batch_cnt])
        else:
            self.batch_cnt = 0
            return (self.images[start_pt:self.data_size],
                    self.labels[start_pt:self.data_size])


def save_pkl_model(weights, biases, save_dir ,f_name):
    # name = os.path.join(data_path, "cifar-10-batches-py/", filename)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    keys = ['cov1','cov2','fc1','fc2','fc3']
    weights_val = {}
    biases_val = {}
    for key in keys:
        weights_val[key] = weights[key].eval()
        biases_val[key] = biases[key].eval()
    with open(save_dir + f_name, 'wb') as f:
        print('Created a pickle file')
        pickle.dump((weights_val, biases_val), f)

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

    num_channels = 3
    img_size_cropped = 24

    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    return images

def mask_gradients(weights, grads_and_names, weight_masks, biases, biases_mask, WITH_BIASES):
    new_grads = []
    keys = ['cov1', 'cov2', 'fc1', 'fc2', 'fc3']
    for grad,var_name in grads_and_names:
        flag = 0
        index = 0
        for key in keys:
            if (weights[key]== var_name):
                mask = weight_masks[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
            if (biases[key]== var_name and WITH_BIASES == True):
                mask = biases_mask[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
        # if flag is not set
        if (flag == 0):
            new_grads.append((grad,var_name))
    return new_grads

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

# def compute_file_name(thresholds):
#     keys_cov = ['cov1', 'cov2']
#     keys_fc = ['fc1', 'fc2', 'fc3']
#     name = ''
#     for key in keys_cov:
#         name += 'cov'+ str(int(thresholds[key]*10))
#     for key in keys_fc:
#         if (key == 'fc1'):
#             name += 'fc'+ str(int(thresholds[key]*100))
#         else:
#             name += 'fc'+ str(int(thresholds[key]*10))
#     return name
def compute_file_name(p):
    name = ''
    name += 'cov' + str(int(round(p['cov1'] * 10)))
    name += 'cov' + str(int(round(p['cov2'] * 10)))
    name += 'fc' + str(int(round(p['fc1'] * 100)))
    name += 'fc' + str(int(round(p['fc2'] * 10)))
    name += 'fc' + str(int(round(p['fc3'] * 10)))
    return name


def main(argv = None):
    if (argv is None):
        argv = sys.argv
    try:
        try:
            opts = argv
            first_time_load = True
            parent_dir = './'
            keys = ['cov1', 'cov2', 'fc1', 'fc2', 'fc3']
            prune_thresholds = {}
            WITH_BIASES = False
            save_for_next_iter = False
            for key in keys:
                prune_thresholds[key] = 0.

            for item in opts:
                print (item)
                opt = item[0]
                val = item[1]
                if (opt == '-cRates'):
                    cRates = val
                if (opt == '-first_time'):
                    first_time_load = val
                if (opt == '-file_name'):
                    file_name = val
                if (opt == '-train'):
                    TRAIN = val
                if (opt == '-prune'):
                    PRUNE = val
                if (opt == '-parent_dir'):
                    parent_dir = val
                if (opt == '-lr'):
                    lr = val
                if (opt == '-with_biases'):
                    WITH_BIASES = val
                if (opt == '-lambda1'):
                    lambda_1 = val
                if (opt == '-lambda2'):
                    lambda_2 = val
                if (opt == '-save'):
                    save_for_next_iter = val
                if (opt == '-org_file_name'):
                    org_file_name = val


            print('pruning thresholds are {}'.format(prune_thresholds))
        except getopt.error, msg:
            raise Usage(msg)
        NUM_CLASSES = 10
        dropout = 0.5
        BATCH_SIZE = 128
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
        INITIAL_LEARNING_RATE = 0.001
        INITIAL_LEARNING_RATE = lr
        LEARNING_RATE_DECAY_FACTOR = 0.1
        NUM_EPOCHS_PER_DECAY = 350.0
        MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
        DISPLAY_FREQ = 50
        TEST = 1
        TRAIN_OR_TEST = 0
        NUM_CHANNELS = 3
        DOCKER = 0
        if (DOCKER == 1):
            # base_model_name = '/root/data/20170206.pkl'
            # model_name = '/root/data/pruning.pkl'
            # mask_dir = '/root/data/mask.pkl'
            base_model_name = '/root/20170206.pkl'
            model_name = '/root/pruning'
            mask_dir = '/root/mask'
        else:
            mask_dir = parent_dir
            weights_dir = parent_dir
        # model_name = 'test.pkl'
        # model_name = '../tf_official_docker/tmp.pkl'

        file_name_part = compute_file_name(cRates)
        if (save_for_next_iter):
            (weights_mask, biases_mask)= initialize_weights_mask(first_time_load, mask_dir, 'mask'+org_file_name + '.pkl')
        else:
            (weights_mask, biases_mask)= initialize_weights_mask(first_time_load, mask_dir, 'mask'+file_name_part + '.pkl')
        cifar10.maybe_download_and_extract()
        class_names = cifar10.load_class_names()

        if (TRAIN):
            images_train, cls_train, labels_train = cifar10.load_training_data()
            images_test, cls_test, labels_test = cifar10.load_test_data()
            t_data = training_data(images_train, labels_train)
            DATA_CNT = len(images_train)
            NUMBER_OF_BATCH = DATA_CNT / BATCH_SIZE
        else:
            if (PRUNE):
                images_test, cls_test, labels_test = cifar10.load_test_data()
            else:
                images_test, cls_test, labels_test = cifar10.load_test_data()



        training_data_list = []

        if (first_time_load):
            PREV_MODEL_EXIST = 0
            weights, biases = initialize_variables(PREV_MODEL_EXIST, '')
        else:
            PREV_MODEL_EXIST = 1
            if (save_for_next_iter):
                file_name_part = org_file_name
            else:
                file_name_part = compute_file_name(cRates)
            weights, biases = initialize_variables( PREV_MODEL_EXIST,
                                                    weights_dir + 'weights' + file_name_part + '.pkl')

        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.float32, [None, NUM_CLASSES])


        keep_prob = tf.placeholder(tf.float32)
        images = pre_process(x, TRAIN)
        test_images = pre_process(x, False)

        weights_new = {}
        keys = ['cov1', 'cov2', 'fc1', 'fc2', 'fc3']

        for key in keys:
            weights_new[key] = weights[key] * tf.constant(weights_mask[key], dtype=tf.float32)


        l1, l2, pred = cov_network(images, weights_new, biases, keep_prob)
        _, _, test_pred = cov_network(test_images, weights_new, biases, keep_prob)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y)

        correct_prediction = tf.equal(tf.argmax(test_pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # l1_norm = 0. * l1
        # l2_norm = 0. * l2
        l1_norm = lambda_1 * l1
        l2_norm = lambda_2 * l2

        regulization_loss = l1_norm + l2_norm

        opt = tf.train.AdamOptimizer(lr)
        loss_value = tf.reduce_mean(cross_entropy) + regulization_loss
        grads = opt.compute_gradients(loss_value)
        org_grads = [(ClipIfNotNone(grad), var) for grad, var in grads]
        # new_grads = mask_gradients(weights, org_grads, weights_mask, biases, biases_mask, WITH_BIASES)

        # Apply gradients.
        train_step = opt.apply_gradients(org_grads)


        init = tf.global_variables_initializer()
        accuracy_list = np.zeros(5)
        # accuracy_list = np.zeros(5)
        train_acc_list = []
        # Launch the graph
        print('Graph launching ..')
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = (0.7)

        with tf.Session() as sess:
            sess.run(init)

            keys = ['cov1', 'cov2', 'fc1', 'fc2', 'fc3']

            print('pre train pruning info')
            prune_info(weights_new, 0)
            print(78*'-')
            start = time.time()
            if TRAIN == 1:
                # for i in range(0,60000):
                # for i in range(0,6):
                for i in range(0,20000):
                    (batch_x, batch_y) = t_data.feed_next_batch(BATCH_SIZE)
                    train_acc, cross_en, l1_loss, l2_loss= sess.run([accuracy, loss_value, l1_norm, l2_norm], feed_dict = {
                                    x: batch_x,
                                    y: batch_y,
                                    keep_prob: 1.0})
                    if (i % DISPLAY_FREQ == 0):
                        print('This is the {}th iteration of {}pruning, time is {}'.format(
                            i,
                            cRates,
                            time.time() - start
                        ))
                        print("accuracy is {} and cross entropy is {}, l1 loss is {}, l2 loss is {}".format(
                            train_acc,
                            cross_en,
                            l1_loss,
                            l2_loss
                        ))
                        # accuracy_list = np.concatenate((np.array([train_acc]),accuracy_list[0:49]))
                        # accuracy_list = np.concatenate((np.array([train_acc]),accuracy_list[0:19]))
                        accuracy_list = np.concatenate((np.array([train_acc]),accuracy_list[0:4]))
                        if (i%(DISPLAY_FREQ*50) == 0 and i != 0 ):
                            prune_info(weights_new, 0)
                            train_acc_list.append(train_acc)
                            file_name_part = compute_file_name(cRates)
                        if ((np.mean(accuracy_list) > 0.81 and train_acc >= 0.83) or train_acc>=0.85):
                            # accuracy_list = np.zeros(20)
                            test_acc = sess.run(accuracy, feed_dict = {
                                                    x: images_test,
                                                    y: labels_test,
                                                    keep_prob: 1.0})
                            prune_info(weights_new, 0)
                            print('test accuracy is {}'.format(test_acc))
                        # if (np.mean(train_acc) > 0.5):
                            if (test_acc >= 0.82):
                                print("save the network only if it breaks thresholds")
                                save_pkl_model(weights, biases, weights_dir, 'weightssave' + file_name_part + '.pkl')
                                print("training accuracy is large, show the list: {}".format(accuracy_list))
                                # break
                    _ = sess.run(train_step, feed_dict = {
                                    x: batch_x,
                                    y: batch_y,
                                    keep_prob: dropout})

            test_acc = sess.run(accuracy, feed_dict = {
                                    x: images_test,
                                    y: labels_test,
                                    keep_prob: 1.0})
            print("test accuracy is {}".format(test_acc))
            if (save_for_next_iter):
                print('saving for the next iteration of dynamic surgery')
                file_name_part = compute_file_name(cRates)
                file_name = 'weights'+ file_name_part+'.pkl'
                # save_pkl_model(weights, biases, parent_dir, file_name)

                file_name_part = compute_file_name(cRates)
                with open(parent_dir + 'mask' + file_name_part + '.pkl','wb') as f:
                    pickle.dump((weights_mask, biases_mask),f)
            if (TRAIN):
                file_name_part = compute_file_name(cRates)
                # save_pkl_model(weights, biases, weights_dir, 'weights' + file_name_part + '.pkl')
                with open(parent_dir + 'training_data'+file_name_part+'.pkl', 'wb') as f:
                    pickle.dump(train_acc_list, f)

            if (PRUNE):
                print('saving pruned model ...')
                f_name = compute_file_name(cRates)
                prune_weights(  cRates,
                                weights,
                                weights_mask,
                                biases,
                                biases_mask,
                                mask_dir,
                                'mask' + f_name + '.pkl')
                file_name_part = compute_file_name(cRates)
                save_pkl_model(weights, biases, weights_dir, 'weights' + file_name_part + '.pkl')
            return test_acc
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return
if __name__ == '__main__':
    sys.exit(main())
