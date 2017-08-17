import cPickle
import numpy as np
import os
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import vgg as vgg19


def load_dataset(filepath):
    print 'Loading data file %s' % filepath
    with open(filepath, 'rb') as fd:
        cifar10 = cPickle.load(fd)
    image = cifar10['data'].astype(dtype=np.uint8)
    image = image.reshape((-1, 3, 32, 32))
    label = np.asarray(cifar10['labels'], dtype=np.uint8)
    label = label.reshape(label.size, 1)
    return image, label


def load_train_data(dir_path, num_batches=5):
    labels = []
    batchsize = 10000
    images = np.empty((num_batches * batchsize, 3, 32, 32), dtype=np.uint8)
    for did in range(1, num_batches + 1):
        fname_train_data = dir_path + "/data_batch_{}".format(did)
        image, label = load_dataset(fname_train_data)
        images[(did - 1) * batchsize:did * batchsize] = image
        labels.extend(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def load_test_data(dir_path):
    images, labels = load_dataset(dir_path + "/test_batch")
    return np.array(images,  dtype=np.float32), np.array(labels, dtype=np.int32)

def normalize_for_vgg(train_x, test_x):
    mean = train_x.mean()
    std = train_x.std()
    train_x -= mean
    test_x -= mean
    train_x /= std
    test_x /= std
    return train_x, test_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dcnn for cifar10')
    parser.add_argument('model', choices=['vgg', 'alexnet', 'resnet', 'caffe'],
            default='alexnet')
    parser.add_argument('data', default='cifar-10-batches-py')
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()
    assert os.path.exists(args.data), \
        'Pls download the cifar10 dataset via "download_data.py py"'
    print 'Loading data ..................'
    train_x, train_y = load_train_data(args.data)
    test_x, test_y = load_test_data(args.data)

    if args.model == 'vgg':
        train_x, test_x = normalize_for_vgg(train_x, test_x)
        with tf.device('/cpu:0'):
            sess = tf.Session()

            images = tf.placeholder(tf.float32, [1, 32, 32, 3])
            label = tf.placeholder(tf.float32, [1, 10])

            vgg = vgg19.Net()
            vgg.build(images)

            train_data = np.transpose(train_x[0].reshape((1, 3, 32, 32)), (0,2,3,1))
            # print train_data.shape
            # print(vgg.get_var_count())
            sess.run(tf.global_variables_initializer())
            prob = sess.run(vgg.prob, feed_dict={images:train_data})

            print prob

            train_label = [1 if i == train_y[0]-1 else 0 for i in range(10)]
            print train_label

            loss = tf.reduce_sum((vgg.prob-label) ** 2)
            optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

            for i in range(100):
                sess.run(optimizer, feed_dict={images:train_data, label:[train_label]})

            prob = sess.run(vgg.prob, feed_dict={images:train_data})
            print prob
            # print train_y[0]

        # net = vgg19.net()
        # train((train_x, train_y, test_x, test_y), net, 250, vgg_lr, 0.0005,
            #   use_cpu=args.use_cpu)
    else:
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        net = resnet.create_net(args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 200, resnet_lr, 1e-4,
              use_cpu=args.use_cpu)
