import tensorflow as tf
import numpy as np
from tqdm import tqdm
from image_processing.extract_patches import *
from jpeg import *
from evaluate import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

np.random.seed(seed=0)
tf.set_random_seed(0)

def unet_conv2d_block(input_conv, num_filters, kernel_dim, is_training, use_bn, k_initializer):
    conv = tf.layers.conv2d(inputs=input_conv, filters=num_filters, kernel_size=kernel_dim, strides=1, activation=None, padding='SAME', kernel_initializer=k_initializer)
    
    if use_bn:
        batch = tf.layers.batch_normalization(conv, momentum=0.9, training=is_training)
        conv = tf.nn.leaky_relu(batch, alpha=0.5)
    else:
        conv = tf.nn.leaky_relu(conv, alpha=0.5)

    conv = tf.layers.conv2d(inputs=conv, filters=num_filters, kernel_size=kernel_dim, strides=1, activation=None, padding='SAME', kernel_initializer=k_initializer) 

    if use_bn:
        batch = tf.layers.batch_normalization(conv, momentum=0.9, training=is_training)
        conv = tf.nn.leaky_relu(batch, alpha=0.5)
    else:
        conv = tf.nn.leaky_relu(conv, alpha=0.5)

    return conv

def unet(img_w, img_h, img_c, init_kernel_size=12, batch_norm=True):

    X_placeholder = tf.placeholder(tf.float32, shape=(None, img_w, img_h, img_c))
    Y_placeholder = tf.placeholder(tf.float32, shape=(None, img_w, img_h, img_c))
    lr_placeholder = tf.placeholder(tf.float32)
    train_placeholder = tf.placeholder(tf.bool)

    dct_coefs_map = tf.placeholder(tf.float32, shape=(None, img_w, img_h, 1))

    #xavier initialization
    initializer = tf.contrib.layers.xavier_initializer(seed = 0)

    #96x96
    conv1 = unet_conv2d_block(input_conv=X_placeholder, num_filters=init_kernel_size, kernel_dim=[16,16], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    max1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print(max1.get_shape())
    
    #48x48    
    conv2 = unet_conv2d_block(input_conv=max1, num_filters=init_kernel_size*2, kernel_dim=[3,3], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    max2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print(max2.get_shape())

    #24x24
    conv3 = unet_conv2d_block(input_conv=max2, num_filters=init_kernel_size*4, kernel_dim=[3,3], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    max3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    print(max3.get_shape())

    #12x12
    conv4 = unet_conv2d_block(input_conv=max3, num_filters=init_kernel_size*8, kernel_dim=[3,3], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    max4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    print(max4.get_shape())

    #6x6
    conv5 = unet_conv2d_block(input_conv=max4, num_filters=init_kernel_size*16, kernel_dim=[3,3], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    max5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
    print(max5.get_shape())

    #3x3
    conv6 = unet_conv2d_block(input_conv=max5, num_filters=init_kernel_size*32, kernel_dim=[3,3], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    print(conv6.get_shape())

    #6x6
    up5 = tf.layers.conv2d_transpose(inputs=conv6, filters=init_kernel_size*16, kernel_size=[3,3], strides=2, activation=tf.nn.relu, padding='SAME', kernel_initializer=initializer)
    up5 = tf.concat([conv5, up5], 3)
    conv5 = unet_conv2d_block(input_conv=up5, num_filters=init_kernel_size*16, kernel_dim=[3,3], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    print(conv5.get_shape())

    #12x12
    up4 = tf.layers.conv2d_transpose(inputs=conv5, filters=init_kernel_size*8, kernel_size=[3,3], strides=2, activation=tf.nn.relu, padding='SAME', kernel_initializer=initializer)
    up4 = tf.concat([conv4, up4], 3)
    conv4 = unet_conv2d_block(input_conv=up4, num_filters=init_kernel_size*8, kernel_dim=[3,3], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    print(conv4.get_shape())

    #24x24
    up3 = tf.layers.conv2d_transpose(inputs=conv4, filters=init_kernel_size*4, kernel_size=[3,3], strides=2, activation=tf.nn.relu, padding='SAME', kernel_initializer=initializer)
    up3 = tf.concat([conv3, up3], 3)
    conv3 = unet_conv2d_block(input_conv=up3, num_filters=init_kernel_size*4, kernel_dim=[3,3], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    print(conv3.get_shape())

    #48x48
    up2 = tf.layers.conv2d_transpose(inputs=conv3, filters=init_kernel_size*2, kernel_size=[3,3], strides=2, activation=tf.nn.relu, padding='SAME', kernel_initializer=initializer)
    up2 = tf.concat([conv2, up2], 3)
    conv2 = unet_conv2d_block(input_conv=up2, num_filters=init_kernel_size*2, kernel_dim=[3,3], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    print(conv2.get_shape())

    #96x96
    up1 = tf.layers.conv2d_transpose(inputs=conv2, filters=init_kernel_size, kernel_size=[3,3], strides=2, activation=tf.nn.relu, padding='SAME', kernel_initializer=initializer)
    #up1 = tf.concat([conv1, up1, dct_coefs_map], 3)
    up1 = tf.concat([conv1, up1], 3)
    conv1 = unet_conv2d_block(input_conv=up1, num_filters=init_kernel_size, kernel_dim=[8,8], is_training=train_placeholder, use_bn=batch_norm, k_initializer=initializer)
    print(conv1.get_shape())

    output_layer = tf.layers.conv2d(inputs=conv1, filters=img_c, kernel_size=[1,1], strides=1, activation=None, padding = 'SAME', kernel_initializer=initializer)

    #output_layer = output_layer + X_placeholder 

    #mse
    loss = tf.reduce_mean(tf.square( tf.subtract(output_layer,Y_placeholder)))
   
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt = tf.train.AdamOptimizer(learning_rate=lr_placeholder).minimize(loss)

    return X_placeholder, Y_placeholder, dct_coefs_map, lr_placeholder, train_placeholder, output_layer, loss, opt

def conv_layer(conv_input, n_filters, kernel, stride, initializer, train_placeholder):
    conv1 = tf.layers.conv2d(inputs=conv_input, filters=n_filters, kernel_size=kernel, strides=1, activation= tf.nn.relu, padding='SAME', kernel_initializer=initializer)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=n_filters, kernel_size=kernel, strides=stride, activation=None, padding='SAME', kernel_initializer=initializer)
    #batch = tf.layers.batch_normalization(conv, training=train_placeholder)
    relu = tf.nn.relu(conv2)
    return relu

def t_conv_layer(conv_input, n_filters, kernel, stride, initializer, train_placeholder):
    conv1 = tf.layers.conv2d(inputs=conv_input, filters=n_filters, kernel_size=kernel, strides=1, activation= tf.nn.relu, padding='SAME', kernel_initializer=initializer)
    conv2 = tf.layers.conv2d_transpose(inputs=conv1, filters=n_filters, kernel_size=kernel, strides=stride, activation=None, padding='SAME', kernel_initializer=initializer)
    #batch = tf.layers.batch_normalization(conv, training=train_placeholder)
    relu = tf.nn.relu(conv2)
    return relu

def dnCNN_layer(conv_input, n_filters, initializer, train_placeholder):
    conv = tf.layers.conv2d(inputs=conv_input, filters=n_filters, kernel_size=[8,8], strides=1, activation=None, padding = 'SAME', kernel_initializer=initializer)
    batch = tf.layers.batch_normalization(conv, training=train_placeholder)
    relu = tf.nn.relu(batch)
    return relu

def build_dnCNN(img_w, img_h, img_c):

    X_placeholder = tf.placeholder(tf.float32, shape=(None, img_w, img_h, img_c))
    
    Y_placeholder = tf.placeholder(tf.float32, shape=(None, img_w, img_h, img_c))

    lr_placeholder = tf.placeholder(tf.float32)

    train_placeholder = tf.placeholder(tf.bool)

    #xavier initialization
    initializer = tf.contrib.layers.xavier_initializer(seed = 0)

    conv1 = tf.layers.conv2d(inputs=X_placeholder, filters=32, kernel_size=[8,8], strides=1, activation=tf.nn.relu, padding = 'SAME', kernel_initializer=initializer)

    conv2 = dnCNN_layer(conv1, 64, initializer, train_placeholder)

    conv3 = dnCNN_layer(conv2, 64, initializer, train_placeholder)

    conv4 = dnCNN_layer(conv3, 64, initializer, train_placeholder)

    conv5 = dnCNN_layer(conv4, 64, initializer, train_placeholder)
    
    #conv6 = dnCNN_layer(conv5, 64, initializer, train_placeholder)

    #conv7 = dnCNN_layer(conv6, 64, initializer, train_placeholder)
    
    #conv8 = dnCNN_layer(conv7, 64, initializer, train_placeholder)

    #conv9 = dnCNN_layer(conv8, 64, initializer, train_placeholder)

    #conv10 = dnCNN_layer(conv9, 64, initializer, train_placeholder)

    #conv11 = dnCNN_layer(conv10, 64, initializer, train_placeholder)
    
    output_layer = tf.layers.conv2d(inputs=conv5, filters=img_c, kernel_size=[8,8], strides=1, activation=None, padding = 'SAME', kernel_initializer=initializer)

    #mse
    loss = tf.reduce_mean(tf.square(output_layer - Y_placeholder))

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt = tf.train.AdamOptimizer(learning_rate=lr_placeholder).minimize(loss)

    return X_placeholder, Y_placeholder, lr_placeholder, train_placeholder, output_layer, loss, opt



batch = extract_patches("images/frame1.jpeg")

dct100, dct10 = get_patches_dcts(batch)

#print(w_image)

#learning_rate = 0.001
learning_rate = 0.1

train = True



sess = tf.InteractiveSession()

X_placeholder, Y_placeholder, dct_coefs_map, lr_placeholder, train_placeholder, output_layer, loss, opt = unet(img_w=IMG_DEFAULT_SIZE, img_h=IMG_DEFAULT_SIZE, img_c=1, init_kernel_size=16, batch_norm=True)

sess.run(tf.global_variables_initializer())

epochs=200

saver = tf.train.Saver()
#saver.restore(sess, "weights/model_2.ckpt")


best_quality = 0

if train == True:
    for i in range(epochs):

        _, mb_erro = sess.run([opt,loss], feed_dict={X_placeholder: dct10[:,:,:,:1], Y_placeholder: dct100[:,:,:,:1], lr_placeholder: learning_rate, train_placeholder: True})
           
        print("train:", "epoch", i, "mean batch error:", mb_erro)

        if i > 0 and i%5 == 0:
            
            predict = sess.run(output_layer, feed_dict={X_placeholder: dct10[:,:,:,:1], train_placeholder: False})
            quality = evaluate_model(dct10.copy(), dct100.copy(), predict.copy(), write_out=False) 
            if quality > best_quality and quality > 0.75:
                best_quality = quality
                save_path = saver.save(sess, "weights/model_2.ckpt")
                print("new best model saved at ...", save_path)
'''
else:

    (minibatch_X, minibatch_Y) = minibatches_test[0]
    #minibatch_X = minibatch_X[:1]
    #minibatch_Y = minibatch_Y[:1]
    predict = sess.run(output_layer, feed_dict={X_placeholder: minibatch_X[:,:,:,:1], train_placeholder: False})
    evaluate_model(minibatch_X.copy(), minibatch_Y.copy(), predict.copy(), write_out=True)
'''
 