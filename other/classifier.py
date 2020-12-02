# _*_ coding:utf-8 _*_
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

np.random.seed(1337)

# 下载mnist数据集(0-9的手写数字)
# 图片的大小28*28
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据预处理
X_train = X_train.reshape(X_train.shape[0], -1)/255  # 变成0-1之间
X_test = X_test.reshape(X_test.shape[0], -1)/255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 另一种构建神经网络的方法
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

# 另一种方法定义优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

print("Training---------")
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)

print('\nTesting----------')
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss:', loss)
print('test accuracy:', accuracy)
