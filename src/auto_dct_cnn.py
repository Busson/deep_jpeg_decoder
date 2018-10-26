import tensorflow as tf
import numpy as np
from dataset_loader import load_dcts
from jpeg import *
from evaluate import *

np.random.seed(seed=0)
tf.set_random_seed(0)


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

    conv2 = dnCNN_layer(conv1, 32, initializer, train_placeholder)

    conv3 = dnCNN_layer(conv2, 32, initializer, train_placeholder)

    conv4 = dnCNN_layer(conv3, 32, initializer, train_placeholder)

    conv5 = dnCNN_layer(conv4, 32, initializer, train_placeholder)

    output_layer = tf.layers.conv2d(inputs=conv5, filters=img_c, kernel_size=[8,8], strides=1, activation=None, padding = 'SAME', kernel_initializer=initializer)

    #mse
    loss = tf.reduce_mean(tf.square(output_layer - Y_placeholder))

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt = tf.train.AdamOptimizer(learning_rate=lr_placeholder).minimize(loss)

    return X_placeholder, Y_placeholder, lr_placeholder, train_placeholder, output_layer, loss, opt



dct_100_1, dct_10_1 = load_dcts("images/parrot.bmp", ".bmp")

x_batch = np.zeros((2,96,96,1))
y_batch = np.zeros((2,96,96,1))

x_batch[0,:,:,0] = dct_10_1[:,:,0]
y_batch[0,:,:,0] = dct_100_1[:,:,0] - dct_10_1[:,:,0]

dct_100_2, dct_10_2 = load_dcts("images/lena_96.bmp", ".bmp")

x_batch[1,:,:,0] = dct_10_2[:,:,0]
y_batch[1,:,:,0] = dct_100_2[:,:,0] - dct_10_2[:,:,0]



learning_rate = 0.001

train = False

sess = tf.InteractiveSession()

X_placeholder, Y_placeholder, lr_placeholder, train_placeholder, output_layer, loss, opt = build_dnCNN(96, 96, 1)

sess.run(tf.global_variables_initializer())

epochs=1000

saver = tf.train.Saver()
saver.restore(sess, "weights3/model_1.ckpt")

if train == True:
    for i in range(epochs):
        _, mb_erro = sess.run([opt,loss], feed_dict={X_placeholder: x_batch, Y_placeholder: y_batch, lr_placeholder: learning_rate, train_placeholder: True})
        print("epoch:", i,"erro:", mb_erro)
    
    save_path = saver.save(sess, "weights3/model_1.ckpt")
    print("new best model saved at ...", save_path)
else:
    #saver.restore(sess, "weights/model_1.ckpt")
    output_model = sess.run(output_layer, feed_dict={X_placeholder: x_batch, train_placeholder: False})

    #view - debug
    qtable_luma_100, qtable_chroma_100 = generate_qtables(quality_factor=100)
    qtable_luma_10, qtable_chroma_10 = generate_qtables(quality_factor=10)

    dct_sp_0 = np.zeros((96,96,3))
    #print(output_model.shape)
    dct_sp_0[:,:,0] = output_model[0,:,:,0] + dct_10_1[:,:,0]
    dct_sp_0[:,:,1] = dct_10_1[:,:,1]
    dct_sp_0[:,:,2] = dct_10_1[:,:,2]

    dct_sp_1 = np.zeros((96,96,3))
    #print(output_model.shape)
    dct_sp_1[:,:,0] = output_model[1,:,:,0] + dct_10_2[:,:,0]
    dct_sp_1[:,:,1] = dct_10_2[:,:,1]
    dct_sp_1[:,:,2] = dct_10_2[:,:,2]

    dec_img_100_0 = decode_image(dct_100_1, qtable_luma_100, qtable_chroma_100)
    dec_img_100_1 = decode_image(dct_100_2, qtable_luma_100, qtable_chroma_100)
    dec_img_10_0 = decode_image(dct_10_1, qtable_luma_10, qtable_chroma_10)
    dec_img_10_1 = decode_image(dct_10_2, qtable_luma_10, qtable_chroma_10)
    dec_img_sp_0 = decode_image(dct_sp_0, qtable_luma_100, qtable_chroma_10)
    dec_img_sp_1 = decode_image(dct_sp_1, qtable_luma_100, qtable_chroma_10)


    print("100 - NRMSE:", calc_nrmse(dec_img_100_1, dec_img_100_1), 
    "SSIM:", calc_ssim(dec_img_100_1, dec_img_100_1),
    "PSNR:", calc_psnr(dec_img_100_1, dec_img_100_1))

    print("10 - NRMSE:", calc_nrmse(dec_img_100_1, dec_img_10_1), 
    "SSIM:", calc_ssim(dec_img_100_1, dec_img_10_1),
    "PSNR:", calc_psnr(dec_img_100_1, dec_img_10_1))

    print("PR - NRMSE:", calc_nrmse(dec_img_100_1, dec_img_sp_1), 
    "SSIM:", calc_ssim(dec_img_100_1, dec_img_sp_1),
    "PSNR:", calc_psnr(dec_img_100_1, dec_img_sp_1))

    print(output_model[0,:,:,0] + dct_10_2[:,:,0])
    print(dct_100_2[:,:,0])

    
    stack0 = np.hstack([dec_img_100_0, dec_img_10_0, dec_img_sp_0])
    stack1 = np.hstack([dec_img_100_1, dec_img_10_1, dec_img_sp_1])
    stack = np.vstack([stack0,stack1])

    cv2.imshow('image',stack)
    cv2.waitKey(0)
