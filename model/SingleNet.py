import os
import sys
import tensorflow as tf
import numpy as np
from Dataset import DataSet

ROOTDIR = os.path.abspath(os.path.join(sys.path[0]))
sys.path.append(ROOTDIR)
tf.compat.v1.disable_eager_execution()
STEPS = 4000
LEARNING_RATE = 0.0001
CLASS_NUM = 4

## read numpy data
def load_data(filename):
    try:
        data = np.load(filename)
        print('load_data', filename)
        print('images', data['images'].shape)
        print('features', data['features'].shape)
        print('labels', data['labels'].shape)
        ds = DataSet(data['images'], data['features'], data['labels'], sample='density')
    except:
        print("Can not find data file")
        ds = None
    finally:
        return ds

# help functions to build graph
def weight_variable(shape):
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.compat.v1.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x):
    return tf.compat.v1.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def single_net(RES):
    with tf.name_scope('input'):
        x = tf.compat.v1.placeholder(tf.float32, shape=[None, RES, RES], name='x')
        ft = tf.compat.v1.placeholder(tf.float32, shape=[None, 14], name='ft')
        y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, CLASS_NUM], name='y')

        x_image = tf.reshape(x, [-1, RES, RES, 1], name='x-reshape')

    # first layer
    with tf.name_scope('layer1'):
        W_conv1 = weight_variable([3, 3, 1, 16])
        b_conv1 = bias_variable([16])

        h_conv1 = tf.compat.v1.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        h_pool1 = max_pool_2x2(h_conv1)
        # [-1, 64, 64, 16]

    # second layer
    with tf.name_scope('layer2'):
        W_conv2 = weight_variable([3, 3, 16, 32])
        b_conv2 = bias_variable([32])

        h_conv2 = tf.compat.v1.nn.relu(conv2d(h_pool1, W_conv2, strides=[1, 2, 2, 1]) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        # [-1, 16, 16, 32]

    with tf.name_scope('layer3'):
        W_conv3 = weight_variable([3, 3, 32, 64])
        b_conv3 = bias_variable([64])

        h_conv3 = tf.compat.v1.nn.relu(conv2d(h_pool2, W_conv3, strides=[1, 2, 2, 1]) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)
        # [-1, 4, 4, 64] = [-1, 1024]

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4 * 4 * 64, 512])
        b_fc1 = bias_variable([512])

        h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 64])
        h_fc1 = tf.compat.v1.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        # [-1, 512]

    return x, ft, y_, h_fc1

class DLSpMVModel(object):
    def __init__(self, train_data, test_data, model_data, result_data):

        self.RES = 0
        # self.mean = 0
        # self.std = 1

        self.train = load_data(train_data)
        print(self.train._images.shape, self.train.labels.shape)
        if self.train:
            self.RES = self.train._images.shape[1] # 128
            # self.channel_num = self.train._images.shape[-1] #2
            # self.mean = np.mean(self.train._images[:,0,:,:], axis=0)
            # self.std = np.std(self.train._images[:,0,:,:], axis=0)

        self.test = load_data(test_data)
        print(self.test._images.shape, self.test.labels.shape)
        if self.test and self.RES == 0:
            self.RES = self.test._images.shape[-1] # 128
            # self.channel_num = self.train._images.shape[-1] #2

        STEPS = 4000
        self.output = result_data
        self.model = model_data

    def build_graph(self):
        pass

    def training(self):

        print("Model is in training mode")
        assert self.train is not None and self.test is not None, "data not loaded"

        with tf.name_scope('fc_snip1'):
            x, ft, y_, h_fc1 = single_net(self.RES)
        #[-1, 512]

        with tf.name_scope('dropout'):
            keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
            h_fc1_drop = tf.compat.v1.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([512, 64])
            b_fc2 = bias_variable([64])
            h1_fc2 = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)

        h_fc2 = tf.concat([h1_fc2, ft], axis=1)

        with tf.name_scope('out'):
            W_fc3 = weight_variable([64 + 14, CLASS_NUM])
            b_fc3 = bias_variable([CLASS_NUM])

            y_conv = tf.add(tf.matmul(h_fc2, W_fc3), b_fc3, name='y_conv_restore')

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(
                tf.compat.v1.nn.softmax_cross_entropy_with_logits(
                    labels=y_, logits=y_conv)  # takes unnormalized output
            )

        with tf.name_scope('train'):
            train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='acc_to_restore')
            tf.summary.scalar('accuracy', accuracy)

        merged = tf.compat.v1.summary.merge_all()
        saver = tf.compat.v1.train.Saver()  # traditional saving api

        # train the model
        with tf.compat.v1.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            for i in range(STEPS):
                batch = self.train.next_batch(100)
                if i % 100 == 0:
                    train_accuracy, train_loss = sess.run(
                        [accuracy, cross_entropy],
                        feed_dict={
                            x: batch[0][:,:,:], ft: batch[1], y_: batch[2],
                            keep_prob: 1.0
                        }
                    )
                    print('step', i, 'train', train_accuracy, train_loss)
                else:
                    _ = sess.run(
                        train_step, 
                        feed_dict={
                            x: batch[0][:,:,:], ft: batch[1], y_: batch[2],
                            keep_prob: 0.5
                        }
                    )
            # test
            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: self.test.images[:,:,:], ft: self.test.features, y_: self.test.labels,
                keep_prob: 1.0
            }))

            # save model and checkpoint
            save_path = saver.save(sess, os.path.join(ROOTDIR, self.model, "SingleNet-{}-{}-{}.ckpt".format(CLASS_NUM,STEPS, LEARNING_RATE)))
            print("Model saved in file %s" % save_path)

    def testing(self):
        """ restore model and checkpoint

        [description]
        """
        print("Model is in testing mode")
        assert self.test is not None, "data not loaded"

        tf.reset_default_graph() # the graph is empty now, must build graph before restore value

        with tf.Session() as sess:
            # retore graph
            saver = tf.train.import_meta_graph(os.path.join(ROOTDIR, self.model, "SingleNet-{}-{}-{}.ckpt.meta".format(CLASS_NUM,STEPS, LEARNING_RATE)))
            # the current graph can be explored by
            graph = tf.get_default_graph()
            # restore value
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(ROOTDIR, self.model)))
            print("Model restored")

            x = graph.get_tensor_by_name("fc_snip1/input/x:0")
            ft = graph.get_tensor_by_name("fc_snip1/input/ft:0")
            y_ = graph.get_tensor_by_name("fc_snip1/input/y:0")
            x2 = graph.get_tensor_by_name("fc_snip2/input/x:0")
            ft2 = graph.get_tensor_by_name("fc_snip2/input/ft:0")
            y2_ = graph.get_tensor_by_name("fc_snip2/input/y:0")

            keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
            #acc = graph.get_tensor_by_name('train/acc_to_restore:0')
            y_conv = graph.get_tensor_by_name('out/y_conv_restore:0')
            print("-------------------------------------------------------")
            out_y = sess.run(y_conv, feed_dict={x: self.test.images[:,0,:,:], ft: self.test.features, y_: self.test.labels, x2: self.test.images[:,1,:,:], ft2: self.test.features, y2_: self.test.labels, keep_prob: 1.0})

            wrongIds = np.zeros((self.test.labels.shape[0], 2), dtype='int32')
            for i in range(self.test.labels.shape[0]):
                wrongIds[i][0] = np.argmax(self.test.labels[i])
                wrongIds[i][1] = np.argmax(out_y[i])
            np.savez('{}'.format(self.output), wrongIds=wrongIds)
            print("-------------------------------------------------------")


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("usage: {} flag{train, test} {train data} {test data} {model data} {result data}")
        exit()

    FLAG = sys.argv[1].lower()
    train_data = sys.argv[2]
    test_data = sys.argv[3]
    model_data = sys.argv[4]
    result_data = sys.argv[5]

    print(train_data)
    print(test_data)
    print(model_data)
    print(result_data)

    model = DLSpMVModel(os.path.join(ROOTDIR, train_data),
                        os.path.join(ROOTDIR, test_data),
                        os.path.join(ROOTDIR, model_data),
                        os.path.join(ROOTDIR, result_data))

    if FLAG == 'train':
        model.training()
    elif FLAG == 'test':
        model.testing()

