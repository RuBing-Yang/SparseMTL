import os
import sys
import tensorflow as tf
import numpy as np
from MTLDataset import DataSet

ROOTDIR = os.path.abspath(os.path.join(sys.path[0]))
sys.path.append(ROOTDIR)
tf.compat.v1.disable_eager_execution()

## read numpy data
def load_data(filename):
    try:
        print('load_data', filename)
        data = np.load(filename)
        ds = DataSet([data['images_dim0'], data['images_dim1']], data['features'], data['labels'], sample='histogram')
    except Exception as e:
        print("load_data except:", e)
        ds = None
    finally:
        return ds

# help functions to build graph
def weight_variable(shape, suffix=''):
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.1, name='weight_' + suffix)
    return tf.Variable(initial)


def bias_variable(shape, suffix=''):
    initial = tf.constant(0.1, shape=shape, name='bias_' + suffix)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.compat.v1.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x):
    return tf.compat.v1.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def single_net(RES, CLASS_NUM):
    with tf.name_scope('input'):
        x = tf.compat.v1.placeholder(tf.float32, shape=[None, RES, RES], name='x')
        tf.compat.v1.summary.histogram(name='x', values=x)
        
        ft = tf.compat.v1.placeholder(tf.float32, shape=[None, 14], name='ft')
        y_spmv = tf.compat.v1.placeholder(tf.float32, shape=[None, CLASS_NUM], name='y_spmv')
        y_sddmm = tf.compat.v1.placeholder(tf.float32, shape=[None, CLASS_NUM], name='y_sddmm')

        x_image = tf.reshape(x, [-1, RES, RES, 1], name='x-reshape')

    # first layer
    with tf.name_scope('layer1'):
        W_conv1 = weight_variable([3, 3, 1, 16], 'conv1')
        x_conv1 = conv2d(x_image, W_conv1)

        tf.compat.v1.summary.histogram(name='x_conv1', values=x_conv1)
        # x_norm1 = tf.compat.v1.nn.batch_normalization(x_conv1, 0, 1, None, None, 1e-12)

        b_conv1 = bias_variable([16], 'conv1')
        h_conv1 = tf.compat.v1.nn.relu(x_conv1 + b_conv1)

        h_pool1 = max_pool_2x2(h_conv1)
        # [-1, 64, 64, 16]

    # second layer
    with tf.name_scope('layer2'):
        W_conv2 = weight_variable([3, 3, 16, 32], 'conv2')
        x_conv2 = conv2d(h_pool1, W_conv2, strides=[1, 2, 2, 1])

        tf.compat.v1.summary.histogram(name='x_conv2', values=x_conv2)
        # x_norm2 = tf.compat.v1.nn.batch_normalization(x_conv2, 0, 1, None, None, 1e-12)

        b_conv2 = bias_variable([32], 'conv2')
        h_conv2 = tf.compat.v1.nn.relu(x_conv2 + b_conv2)

        h_pool2 = max_pool_2x2(h_conv2)
        # [-1, 16, 16, 32]

    with tf.name_scope('layer3'):
        W_conv3 = weight_variable([3, 3, 32, 64], 'conv3')
        x_conv3 = conv2d(h_pool2, W_conv3, strides=[1, 2, 2, 1])
    
        tf.compat.v1.summary.histogram(name='x_conv3', values=x_conv3)
        # x_norm3 = tf.compat.v1.nn.batch_normalization(x_conv3, 0, 1, None, None, 1e-12)
    
        b_conv3 = bias_variable([64], 'conv3')
        h_conv3 = tf.compat.v1.nn.relu(x_conv3 + b_conv3)
        
        h_pool3 = max_pool_2x2(h_conv3)
        # [-1, 4, 4, 64] = [-1, 1024]

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4 * 4 * 64, 512], 'fc1')
        b_fc1 = bias_variable([512], 'fc1')

        h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 64])
        h_fc1 = tf.compat.v1.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        # [-1, 512]

    return x, ft, y_spmv, y_sddmm, h_fc1

class DLSpMVModel(object):
    def __init__(self, train_data, test_data, model_data, result_data):

        self.RES = 0
        self.STEPS = 8000
        self.LEARNING_RATE = 0.0001
        self.CLASS_NUM = 4
        self.FOLDER = '/map'
        # self.mean = 0
        # self.std = 1

        self.train = load_data(train_data)
        self.CLASS_NUM = self.train.labels0.shape[1]
        print(self.train._images_dim0.shape, self.train.labels0.shape)
        if self.train:
            self.RES = self.train._images_dim0.shape[1] # 128
            # self.channel_num = self.train._images_dim0.shape[-1] #2
            # self.mean = np.mean(self.train._images[:,0,:,:], axis=0)
            # self.std = np.std(self.train._images[:,0,:,:], axis=0)

        self.test = load_data(test_data)
        print(self.test._images_dim0.shape, self.test.labels0.shape)
        if self.test and self.RES == 0:
            self.RES = self.test._images_dim0.shape[-1] # 128
            # self.channel_num = self.train._images_dim0.shape[-1] #2
        self.output = result_data
        self.model = model_data

    def build_graph(self):
        pass

    def training(self):

        print("Model is in training mode")
        assert self.train is not None and self.test is not None, "data not loaded"

        with tf.name_scope('fc_snip1'):
            x, ft, y_spmv, y_sddmm, h_fc1_snip1 = single_net(self.RES, self.CLASS_NUM)
            tf.compat.v1.summary.histogram(name='y_spmv', values=y_spmv)
            tf.compat.v1.summary.histogram(name='y_sddmm', values=y_sddmm)
        #[-1, 512]

        with tf.name_scope('fc_snip2'):
            x2, ft2, y2_spmv, y2_sddmm, h_fc1_snip2 = single_net(self.RES, self.CLASS_NUM)

        h_fc1 = tf.concat([h_fc1_snip1, h_fc1_snip2], axis=1)
        # [-1, 512 * 2]

        with tf.name_scope('dropout'):
            keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
            h_fc1_drop = tf.compat.v1.nn.dropout(h_fc1, keep_prob)
            tf.compat.v1.summary.histogram(name='h_fc1_drop', values=h_fc1_drop)

        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([512 * 2, 64], 'fc2')
            b_fc2 = bias_variable([64], 'fc2')
            h1_fc2 = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)

        h_fc2 = tf.concat([h1_fc2, ft], axis=1)

        with tf.name_scope('out'):
            W_fc3_spmv = weight_variable([64 + 14, self.CLASS_NUM], 'fc3_spmv')
            b_fc3_spmv = bias_variable([self.CLASS_NUM], 'fc3_spmv')
            y_conv_spmv = tf.add(tf.matmul(h_fc2, W_fc3_spmv), b_fc3_spmv, name='y_conv_spmv')

            W_fc3_sddmm = weight_variable([64 + 14, self.CLASS_NUM], 'fc3_sddmm')
            b_fc3_sddmm = bias_variable([self.CLASS_NUM], 'fc3_sddmm')
            y_conv_sddmm = tf.add(tf.matmul(h_fc2, W_fc3_sddmm), b_fc3_sddmm, name='y_conv_sddmm')

        with tf.name_scope('loss'):
            cross_entropy_spmv = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_spmv, logits=y_conv_spmv)  # takes unnormalized output
            )
            cross_entropy_sddmm = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_sddmm, logits=y_conv_sddmm)  # takes unnormalized output
            )
            loss = cross_entropy_spmv + cross_entropy_sddmm
            tf.compat.v1.summary.scalar(name='cross_entropy_spmv', tensor=cross_entropy_spmv)
            tf.compat.v1.summary.scalar(name='cross_entropy_sddmm', tensor=cross_entropy_sddmm)

        with tf.name_scope('evaluate'):
            train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(loss)

            correct_pred_spmv = tf.equal(tf.argmax(y_conv_spmv, 1), tf.argmax(y_spmv, 1))
            accuracy_spmv = tf.reduce_mean(tf.cast(correct_pred_spmv, tf.float32))
            tf.compat.v1.summary.scalar(name='accuracy_spmv', tensor=accuracy_spmv)
            
            correct_pred_sddmm = tf.equal(tf.argmax(y_conv_sddmm, 1), tf.argmax(y_sddmm, 1))
            accuracy_sddmm = tf.reduce_mean(tf.cast(correct_pred_sddmm, tf.float32))
            tf.compat.v1.summary.scalar(name='accuracy_sddmm', tensor=accuracy_sddmm)

        saver = tf.compat.v1.train.Saver()  # traditional saving api
        train_writer = tf.compat.v1.summary.FileWriter('./logs' + self.FOLDER+ '/spmv-sddmm-active-{}-{}-{}/train'.format(self.CLASS_NUM, self.STEPS, self.LEARNING_RATE))
        test_writer = tf.compat.v1.summary.FileWriter('./logs' + self.FOLDER+ '/spmv-sddmm-active-{}-{}-{}/test'.format(self.CLASS_NUM, self.STEPS, self.LEARNING_RATE))
        merged_summary = tf.compat.v1.summary.merge_all()
        # train_writer = tf.summary.create_file_writer('./logs')

        # train the model
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            # train_writer = tf.compat.v1.summary.FileWriter('./logs', graph=sess.graph)
            for i in range(self.STEPS):
                batch = self.train.next_batch(100)
                if i % 100 == 0:
                    train_summary, train_accuracy_spmv, train_accuracy_sddmm, train_loss_spmv, train_loss_sddmm = sess.run(
                        [merged_summary, accuracy_spmv, accuracy_sddmm, cross_entropy_spmv, cross_entropy_sddmm], 
                        feed_dict={
                            x: batch[0][:,:,:], ft: batch[2], y_spmv: batch[3], y_sddmm: batch[4],
                            x2: batch[1][:,:,:], ft2: batch[2], y2_spmv: batch[3], y2_sddmm: batch[4],
                            keep_prob: 1.0
                        }
                    )
                    train_writer.reopen()
                    train_writer.add_summary(train_summary, global_step=i)
                    train_writer.close()
                    
                    test_summary = merged_summary.eval(feed_dict={
                        x: self.test.images_dim0[:,:,:], ft: self.test.features, y_spmv: self.test.labels0, y_sddmm: self.test.labels1, 
                        x2: self.test.images_dim1[:,:,:], ft2: self.test.features, y2_spmv: self.test.labels0, y2_sddmm: self.test.labels1, 
                        keep_prob: 1.0
                    })
                    test_writer.reopen()
                    test_writer.add_summary(test_summary, global_step=i)
                    test_writer.close()
                    print('step %d, training accuracy=%g %g loss=%g %g' % (i, train_accuracy_spmv, train_accuracy_sddmm, train_loss_spmv, train_loss_sddmm))
                else:
                    _ = sess.run(
                        train_step, 
                        feed_dict={
                            x: batch[0][:,:,:], ft: batch[2], y_spmv: batch[3], y_sddmm: batch[4],
                            x2: batch[1][:,:,:], ft2: batch[2], y2_spmv: batch[3], y2_sddmm: batch[4],
                            keep_prob: 0.5
                        }
                    )
            # test
            print('test accuracy %g %g' % (
                accuracy_spmv.eval(feed_dict={
                    x: self.test.images_dim0[:,:,:], ft: self.test.features, y_spmv: self.test.labels0, y_sddmm: self.test.labels1, 
                    x2: self.test.images_dim1[:,:,:], ft2: self.test.features, y2_spmv: self.test.labels0, y2_sddmm: self.test.labels1, 
                    keep_prob: 1.0
                }),
                accuracy_sddmm.eval(feed_dict={
                    x: self.test.images_dim0[:,:,:], ft: self.test.features, y_spmv: self.test.labels0, y_sddmm: self.test.labels1, 
                    x2: self.test.images_dim1[:,:,:], ft2: self.test.features, y2_spmv: self.test.labels0, y2_sddmm: self.test.labels1, 
                    keep_prob: 1.0
                })
            ))
            test_writer.reopen()
            test_writer.add_summary(test_summary, global_step=self.STEPS)
            test_writer.close()

            # save model and checkpoint
            save_path = saver.save(sess, os.path.join(ROOTDIR, self.model, "spmv-sddmm-active-{}-{}-{}.ckpt".format(self.CLASS_NUM, self.STEPS, self.LEARNING_RATE)))
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
            saver = tf.train.import_meta_graph(os.path.join(ROOTDIR, self.model, "spmv-sddmm-active-{}-{}-{}.ckpt.meta".format(self.CLASS_NUM, self.STEPS, self.LEARNING_RATE)))
            # the current graph can be explored by
            graph = tf.get_default_graph()
            # restore value
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(ROOTDIR, self.model)))
            print("Model restored")

            x = graph.get_tensor_by_name("fc_snip1/input/x:0")
            ft = graph.get_tensor_by_name("fc_snip1/input/ft:0")
            y_spmv = graph.get_tensor_by_name("fc_snip1/input/y_spmv:0")
            y_sddmm = graph.get_tensor_by_name("fc_snip1/input/y_spmv:0")

            x2 = graph.get_tensor_by_name("fc_snip2/input/x:0")
            ft2 = graph.get_tensor_by_name("fc_snip2/input/ft:0")
            y2_spmv = graph.get_tensor_by_name("fc_snip2/input/y_spmv:0")
            y2_sddmm = graph.get_tensor_by_name("fc_snip2/input/y_spmv:0")

            keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")
            y_conv = graph.get_tensor_by_name('out/y_conv_restore:0')
            print("-------------------------------------------------------")
            out_y = sess.run(y_conv, feed_dict={
                x: self.test.images[:,0,:,:], ft: self.test.features, y_spmv: self.test.labels0, y_sddmm: self.test.labels1, 
                x2: self.test.images[:,1,:,:], ft2: self.test.features, y2_spmv: self.test.labels0, y2_sddmm: self.test.labels1, 
                keep_prob: 1.0})

            wrongIds = np.zeros((self.test.labels.shape[0], 2), dtype='int32')
            for i in range(self.test.labels.shape[0]):
                wrongIds[i][0] = np.argmax(self.test.labels[i])
                wrongIds[i][1] = np.argmax(out_y[i])
            np.savez('{}'.format(self.output), wrongIds=wrongIds)
            print("-------------------------------------------------------")


def main():
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
    
    model_data_list = model_data.split('/')
    print(model_data_list)
    model_data_list.remove('')
    print(model_data_list)
    model.FOLDER = '/' + model_data_list[-1]
    print('model.CLASS_NUM', model.CLASS_NUM)
    print('model.FOLDER', model.FOLDER)

    if FLAG == 'train':
        model.training()
    elif FLAG == 'test':
        model.testing()

if __name__ == '__main__':
    main()
