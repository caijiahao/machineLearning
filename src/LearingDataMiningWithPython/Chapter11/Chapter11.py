#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/23 14:57
# @Author  : Aries
# @Site    : 
# @File    : Chapter11.py
# @Software: PyCharm

import os
batch1_filename = "data_batch_1"

import pickle
def unpickle(filename):
    with open(filename, 'rb') as fo:
        return pickle.load(fo)

batch1 = unpickle(batch1_filename)
image_index = 100
image = batch1['data'][image_index]

image = image.reshape((32,32, 3), order='F')
import numpy as np
image = np.rot90(image, -1)

#用matplotlib绘制图像
from matplotlib import pyplot as plt
#plt.imshow(image)
#plt.show()

#创建计算直角三角形斜边长度的函数
import theano
from theano import tensor as T
a = T.dscalar()
b = T.dscalar()
c = T.sqrt(a ** 2 + b ** 2)
#定义一个函数
f = theano.function([a,b], c)
print f(3, 4)

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data.astype(np.float32)
y_true = iris.target.astype(np.int32)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true, random_state=14)

import lasagne
#创建输入层
input_layer = lasagne.layers.InputLayer(shape=(10, X.shape[1]))

#创建隐含层
hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=12, nonlinearity=lasagne.nonlinearities.sigmoid)

#创建输入层
output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=3,nonlinearity=lasagne.nonlinearities.softmax)

#为了训练刚创建的网络，我们定义几个Theano训练函数
import theano.tensor as T
#net_input = T.matrix('net_input')
#net_output = output_layer.get_output_for(net_input)
net_output=lasagne.layers.get_output(output_layer)
true_output = T.ivector('true_output')

loss = T.mean(T.nnet.categorical_crossentropy(net_output, true_output))

#获取到网络的所有参数，创建调整权重的函数，使损失降到最低
all_params = lasagne.layers.get_all_params(output_layer)
updates = lasagne.updates.sgd(loss, all_params, learning_rate=0.1)

#训练网络，然后获取网络输出，以用于后续测试
import theano
train = theano.function([input_layer.input_var, true_output], outputs=loss, updates=updates)
get_output = theano.function([input_layer.input_var], net_output)

#调用训练函数
for n in range(1000):
    train(X_train, y_train)

y_output = get_output(X_test)

#找出激励作用最高的神经元，就能得到预测结果
import numpy as np
y_pred = np.argmax(y_output, axis=1)

#计算F1值
#from sklearn.metrics import f1_score
#print(f1_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#用nolearn实现神经网络
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import resize
from skimage import transform as tf
from skimage.measure import label, regionprops
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split

def create_captcha(text, shear=0, size=(100, 24)):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r"Coval.otf", 22)
    draw.text((2, 2), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    return image / image.max()

def segment_image(image):
    labeled_image = label(image > 0)
    subimages = []
    for region in regionprops(labeled_image):
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])
    if len(subimages) == 0:
        return [image, ]
    return subimages
random_state = check_random_state(14)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.5, 0.05)

def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    return create_captcha(letter, shear=shear, size=(20, 20)), letters.index(letter)
dataset, targets = zip(*(generate_sample(random_state) for i in range(3000)))
dataset = np.array(dataset, dtype='float')
targets =  np.array(targets)

onehot = OneHotEncoder()
y = onehot.fit_transform(targets.reshape(targets.shape[0],1))
y = y.todense().astype(np.float32)

dataset = np.array([resize(segment_image(sample)[0], (20, 20)) for sample in dataset])
X = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
X = X / X.max()
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=14)

#创建有输入层，密集隐含层和密集输出层组成的层级结构
from lasagne import layers
layers=[
       ('input', layers.InputLayer),
       ('hidden', layers.DenseLayer),
       ('output', layers.DenseLayer),
       ]

from lasagne import updates
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import sigmoid, softmax

net1 = NeuralNet(layers=layers,
                     input_shape=X.shape,
                     hidden_num_units=100,
                     output_num_units=26,
                     hidden_nonlinearity=sigmoid,
                     output_nonlinearity=softmax,
                     hidden_b=np.zeros((100,), dtype=np.float64),
                     update=updates.momentum,
                     update_learning_rate=0.9,
                     update_momentum=0.1,
                     regression=True,
                     max_epochs=1000,
                     )

#在训练集上训练网络
net1.fit(X_train, y_train)

y_pred = net1.predict(X_test)
y_pred = y_pred.argmax(axis=1)
assert len(y_pred) == len(X_test)
if len(y_test.shape) > 1:
    y_test = y_test.argmax(axis=1)
print(classification_report(y_test, y_pred))