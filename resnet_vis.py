import numpy as np 
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2

import cv2
import math
import re
import os

def plotNNFilter(units):
    w = units.shape[1]
    h = units.shape[2]
    c = units.shape[3]
    c_sqrt = math.ceil(math.sqrt(c))
    img = np.zeros([w*(c_sqrt), h*(c_sqrt)])
    for idx in range(c):
        kernel = units[0,:,:,idx]
        i = idx // (c_sqrt)
        j = idx % c_sqrt
        img[i*w:(i+1)*w, j*h:(j+1)*h] = kernel
    return img

def getActivations(layer, inputs):
    units = sess.run(layer, feed_dict={x: inputs})
    return plotNNFilter(units)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, [1, 224, 224, 3])
    
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        output, layers = resnet_v2.resnet_v2_101(x, is_training=False)
        cnn_layers = []
        pattern = re.compile(".*conv*")
        for k in layers:
            if pattern.match(k):
                cnn_layers.append(k) 
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join('./resnet_v2_101.ckpt'),
        slim.get_model_variables('resnet_v2_101'))
    
    with tf.Session() as sess:
        init_fn(sess)
        for img_path in ['simple.png', 'clutter.png', 'outdoor.png']:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            for layer in cnn_layers:
                img_grid = getActivations(layers[layer], [img])
                cv2.imwrite('%s-%s.png' % (img_path, layer.replace('/', '-')), img_grid)

