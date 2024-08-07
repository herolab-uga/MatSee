#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 15:28:25 2022

@author: sivakr
"""

import tensorlayer as tl
import tensorflow as tf
import numpy as np
from LoadData import LoadData
from tensorlayer.utils import dict_to_one
import time

def get_session(gpu_fraction=0.5):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def minibatches(inputs=None, inputs2=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], inputs2[excerpt], targets[excerpt]

def fit(sess, network, train_op, cost, X_train, X_train2, y_train, x, x_2, y_, acc=None, batch_size=100,
        n_epoch=10, print_freq=1, X_val=None, X_val2=None, y_val=None, eval_train=True,
        tensorboard=False, tensorboard_epoch_freq=5, tensorboard_weight_histograms=True, tensorboard_graph_vis=True):
    assert X_train.shape[0] >= batch_size, "Number of training examples should be bigger than the batch size"
    print("Start training the network ...")
    start_time_begin = time.time()
    tensorboard_train_index, tensorboard_val_index = 0, 0
    for epoch in range(n_epoch):
        start_time = time.time()
        loss_ep = 0; n_step = 0
        for X_train_a, X_train_b, y_train_a in minibatches(X_train, X_train2, y_train,
                                                    batch_size, shuffle=True):
            feed_dict = {x: X_train_a, x_2:X_train_b, y_: y_train_a}
            feed_dict.update( network.all_drop )    # enable noise layers
            loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
            loss_ep += loss
            n_step += 1
        loss_ep = loss_ep/ n_step
        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            if (X_val is not None) and (y_val is not None):
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                if eval_train is True:
                    train_loss, train_acc, n_batch = 0, 0, 0
                    for X_train_a, X_train_b, y_train_a in minibatches(
                                            X_train, X_train2, y_train, batch_size, shuffle=True):
                        dp_dict = dict_to_one( network.all_drop )    # disable noise layers
                        feed_dict = {x: X_train_a, x_2:X_train_b, y_: y_train_a}
                        feed_dict.update(dp_dict)
                        if acc is not None:
                            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                            train_acc += ac
                        else:
                            err = sess.run(cost, feed_dict=feed_dict)
                        train_loss += err;  n_batch += 1
                    print("   train loss: %f" % (train_loss/ n_batch))
                    if acc is not None:
                        print("   train acc: %f" % (train_acc/ n_batch))
                val_loss, val_acc, n_batch = 0, 0, 0
                for X_val_a, X_val_b, y_val_a in minibatches(
                                            X_val, X_val2, y_val, batch_size, shuffle=True):
                    dp_dict = dict_to_one( network.all_drop )    # disable noise layers
                    feed_dict = {x: X_val_a, x_2:X_val_b, y_: y_val_a}
                    feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                        # y_predi = y_predi.append(y_pred)
                        val_acc += ac
                    else:
                        err = sess.run([cost], feed_dict=feed_dict)
                        # y_predi = y_predi.append(y_pred)
                    val_loss += err; n_batch += 1
                print("   val loss: %f" % (val_loss/ n_batch))
                if acc is not None:
                    print("   val acc: %f" % (val_acc/ n_batch))
            else:
                print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))
        print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))
    print("Total training time: %fs" % (time.time() - start_time_begin))

def fusionNet(x, x_dep, y_,reuse=True):
    b_init2 = tf.constant_initializer(value=0.1)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
        
    networkRGB = tl.layers.InputLayer(x, name='input_layerRGB')
    networkDepth = tl.layers.InputLayer(x_dep, name='input_layerDepth')
        
    networkRGB = tl.layers.Conv2dLayer(networkRGB,act = tf.nn.relu,shape = (3, 3, 3, 96),strides = (1, 1, 1, 1), padding='VALID',W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init = tf.constant_initializer(value=0.0), name ='cnn_layer1')
    #print (networkRGB)
    networkRGB = tl.layers.PoolLayer(networkRGB,ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1),padding='VALID',pool = tf.nn.max_pool, name ='pool_layer1',) 
        
    networkDepth = tl.layers.Conv2dLayer(networkDepth,act = tf.nn.relu,shape = (3, 3, 1, 96),strides = (1, 1, 1, 1), padding='VALID',W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init = tf.constant_initializer(value=0.0), name ='cnn_layer1depth')
    networkDepth = tl.layers.PoolLayer(networkDepth,ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1),padding='VALID',pool = tf.nn.max_pool, name ='pool_layer1depth',) 
               
    networkRGB = tl.layers.Conv2dLayer(networkRGB,act = tf.nn.relu,shape = (3, 3, 96, 256),strides = (1, 1, 1, 1), padding='VALID',W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init = tf.constant_initializer(value=0.1), name ='cnn_layer2')
    networkRGB = tl.layers.PoolLayer(networkRGB,ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1),padding='VALID',pool = tf.nn.max_pool, name ='pool_layer2',)
        
    networkDepth = tl.layers.Conv2dLayer(networkDepth,act = tf.nn.relu,shape = (3, 3, 96, 256),strides = (1, 1, 1, 1), padding='VALID',W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init = tf.constant_initializer(value=0.1), name ='cnn_layer2depth')
    networkDepth = tl.layers.PoolLayer(networkDepth,ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1),padding='VALID',pool = tf.nn.max_pool, name ='pool_layer2depth',)

 networkRGB = tl.layers.Conv2dLayer(networkRGB,act = tf.nn.relu,shape = (3, 3, 256, 384),strides = (1, 1, 1, 1), padding='VALID',W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init = tf.constant_initializer(value=0.1), name ='cnn_layer3')
    #print (networkRGB)
    networkRGB = tl.layers.PoolLayer(networkRGB,ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1),padding='VALID',pool = tf.nn.max_pool, name ='pool_layer3',) 
        
    networkDepth = tl.layers.Conv2dLayer(networkDepth,act = tf.nn.relu,shape = (3, 3, 256, 384),strides = (1, 1, 1, 1), padding='VALID',W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init = tf.constant_initializer(value=0.1), name ='cnn_layer3depth')
    networkDepth = tl.layers.PoolLayer(networkDepth,ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1),padding='VALID',pool = tf.nn.max_pool, name ='pool_layer3depth',) 
               
    networkRGB = tl.layers.Conv2dLayer(networkRGB,act = tf.nn.relu,shape = (3, 3, 384, 384),strides = (1, 1, 1, 1), padding='VALID',W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init = tf.constant_initializer(value=0.1), name ='cnn_layer4')
    networkRGB = tl.layers.PoolLayer(networkRGB,ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1),padding='VALID',pool = tf.nn.max_pool, name ='pool_layer4',)
        
    networkDepth = tl.layers.Conv2dLayer(networkDepth,act = tf.nn.relu,shape = (3, 3, 384, 384),strides = (1, 1, 1, 1), padding='VALID',W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init = tf.constant_initializer(value=0.1), name ='cnn_layer4depth')
    networkDepth = tl.layers.PoolLayer(networkDepth,ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1),padding='VALID',pool = tf.nn.max_pool, name ='pool_layer4depth',)

 networkRGB = tl.layers.Conv2dLayer(networkRGB,act = tf.nn.relu,shape = (3, 3, 384, 256),strides = (1, 1, 1, 1), padding='VALID',W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init = tf.constant_initializer(value=0.1), name ='cnn_layer5')
    #print (networkRGB)
    networkRGB = tl.layers.PoolLayer(networkRGB,ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1),padding='VALID',pool = tf.nn.max_pool, name ='pool_layer5',) 
        
    networkDepth = tl.layers.Conv2dLayer(networkDepth,act = tf.nn.relu,shape = (3, 3, 384, 256),strides = (1, 1, 1, 1), padding='VALID',W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init = tf.constant_initializer(value=0.1), name ='cnn_layer5depth')
    networkDepth = tl.layers.PoolLayer(networkDepth,ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1),padding='VALID',pool = tf.nn.max_pool, name ='pool_layer5depth',) 
               
            
    networkRGB = tl.layers.FlattenLayer(networkRGB, name='flatten')
    networkDepth = tl.layers.FlattenLayer(networkDepth, name='flatten')
    network = tl.layers.ConcatLayer([networkRGB, networkDepth], 1, name ='concat_layer')    
    network = tl.layers.DenseLayer(network, 4096, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='fc1')
    network = tl.layers.DenseLayer(network, 4096, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='fc2')
    network = tl.layers.DenseLayer(network, n_units=10, act=None, W_init=W_init2, name='outputs')
    return network

if __name__ == "__main__":
    # load data
    data_loader = LoadData(root_path="/home/herobot/sivakr/Server/datasets/rgbd/")
    all_train_rgb_samples, all_train_depth_samples, all_train_labels, all_test_rgb_samples, all_test_depth_samples, all_test_labels = data_loader.load_data()
    session = get_session()

    # define placeholder
    # tf.disable_eager_execution() 
    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name='x')
    x_depth = tf.placeholder(tf.float32, shape=[None,32,32,1], name='x_depth')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    # define loss
    network = fusionNet(x, x_depth, y_)
    y = network.outputs
    cost = tl.cost.cross_entropy(y, y_, name="cost")
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    print(y_op, correct_prediction)

    # define optimizer
    train_params = network.all_params
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
    # train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
    #                                 epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    # initialize
    tl.layers.initialize_global_variables(session)
    saver = tf.compat.v1.train.Saver()
    # list model info
    # network.print_params()
    # network.print_layers()

    # train and test model
    fit(session, network, train_op, cost, np.array(all_train_rgb_samples), np.array(all_train_depth_samples), np.array(all_train_labels), x, x_depth, y_,
                 acc=acc, batch_size=128, n_epoch=300, print_freq=1,
                 X_val=np.array(all_test_rgb_samples), X_val2=np.array(all_test_depth_samples), y_val=np.array(all_test_labels), eval_train=True)

    saver.save(session, 'rgbd_model')
    session.close()
