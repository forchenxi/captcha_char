# _*_ coding:utf-8 _*_
"""
@time :2020/3/17 19:38
@author :liutengfei
@mail:liutengfei@bertadata.com
@desc: 新闻分类：多分类问题
"""
from keras.datasets import reuters
from keras.utils import to_categorical
import numpy as np
from keras import models
from keras import layers
import matplotlib.pylab as plt


def vectorize_sequences(labels, dimension=10000):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    path='D:/python_project/深度学习/keras_t/codes/reuters.npz', num_words=10000)

# 将训练数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 将训练标签向量化
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
# 模型定义
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# 留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))
history_dict = history.history
loss_values = history_dict["loss"]  # 训练数据的损失
acc_values = history_dict["acc"]  # 训练数据的准确率
val_loss_values = history_dict["val_loss"]  # 验证数据的损失
val_acc_values = history_dict["val_acc"]   # 验证数据的准确率

# epochs = range(1, len(loss_values) + 1)
#
# # 绘制训练损失和验证损失
# plt.plot(epochs, loss_values, 'bo', label="Training loss")
# plt.plot(epochs, val_loss_values, 'b', label="Validation loss")
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # 清空图像
# plt.clf()
# # 绘制训练精度和验证精度
# plt.plot(epochs, acc_values, 'bo', label="Training acc")
# plt.plot(epochs, val_acc_values, 'b', label="Validation acc")
# plt.title("Training and validation accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# 测试数据
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# 预测结果
predictions = model.predict(x_test)
print(predictions[0].shape)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))


