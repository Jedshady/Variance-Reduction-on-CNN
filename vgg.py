# Imports
import numpy as np
import tensorflow as tf

class Net(object):
    def __init__(self, vgg19_npy_path = None, trainable = True):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(self, state_input):
        # input layer
        # self.state_input = tf.placeholder(tf.float32, shape=[1, 3, 32, 32], name='input')

        # conv1
        self.conv1_1 = self.conv_layer(state_input, 3, 64, "conv1_1")
        self.conv1_1_dp = tf.nn.dropout(self.conv1_1, 0.7)

        self.conv1_2 = self.conv_layer(self.conv1_1_dp, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, "pool1")

        # conv2
        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_1_dp = tf.nn.dropout(self.conv2_1, 0.6)

        self.conv2_2 = self.conv_layer(self.conv2_1_dp, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, "pool2")

        # conv3
        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_1_dp = tf.nn.dropout(self.conv3_1, 0.6)

        self.conv3_2 = self.conv_layer(self.conv3_1_dp, 256, 256, "conv3_2")
        self.conv3_2_dp = tf.nn.dropout(self.conv3_2, 0.6)

        self.conv3_3 = self.conv_layer(self.conv3_2_dp, 256, 256, "conv3_3")
        self.conv3_3_dp = tf.nn.dropout(self.conv3_3, 0.6)

        self.conv3_4 = self.conv_layer(self.conv3_3_dp, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, "pool3")

        # conv4
        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_1_dp = tf.nn.dropout(self.conv4_1, 0.6)

        self.conv4_2 = self.conv_layer(self.conv4_1_dp, 512, 512, "conv4_2")
        self.conv4_2_dp = tf.nn.dropout(self.conv4_2, 0.6)

        self.conv4_3 = self.conv_layer(self.conv4_2_dp, 512, 512, "conv4_3")
        self.conv4_3_dp = tf.nn.dropout(self.conv4_3, 0.6)

        self.conv4_4 = self.conv_layer(self.conv4_3_dp, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, "pool4")

        # conv5
        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_1_dp = tf.nn.dropout(self.conv5_1, 0.6)

        self.conv5_2 = self.conv_layer(self.conv5_1_dp, 512, 512, "conv5_2")
        self.conv5_2_dp = tf.nn.dropout(self.conv5_2, 0.6)

        self.conv5_3 = self.conv_layer(self.conv5_2_dp, 512, 512, "conv5_3")
        self.conv5_3_dp= tf.nn.dropout(self.conv5_3, 0.6)

        self.conv5_4 = self.conv_layer(self.conv5_3_dp, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, "pool5")

        # fully connected layer
        self.fc6 =  self.fc_layer(self.pool5, 512, 512, "fc6") # 32->16->8->4->2->1*1*512
        self.fc6_dp = tf.nn.dropout(self.fc6, 0.5)

        self.fc7 = self.fc_layer(self.fc6_dp, 512, 10, "fc7")
        self.prob = tf.nn.softmax(self.fc7, name="prob")

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            conv_filters, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

        # convolution and bias
        conv = tf.nn.conv2d(bottom, conv_filters, [1,1,1,1], padding="SAME")
        conv_with_bias = tf.nn.bias_add(conv, conv_biases)

        # batch normalization
        # mean, var = tf.nn.moments(conv_with_bias, [0])
        # conv_biases_bn = tf.nn.batch_normalization(conv_with_bias, mean, var, None, None, variance_epsilon=1e-3)

        # relu
        relu = tf.nn.relu(conv_with_bias)

        return relu

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()

        return var

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            # batch normalization
            # mean, var = tf.nn.moments(fc, [0])
            # fc_bn = tf.nn.batch_normalization(fc, mean, var, None, None, variance_epsilon=1e-3)

            fc_relu = tf.nn.relu(fc)

            return fc_relu

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
