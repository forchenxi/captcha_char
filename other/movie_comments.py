# _*_ coding:utf-8 _*_
"""
@time :2020/1/21 10:19
@author :liutengfei
@mail:liutengfei@bertadata.com
@desc: 电影评论分类：二分类问题
"""
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pylab as plt
from keras.utils import to_categorical

# 评论（单词序列）已经被转换为整数序列，其中每个整数代表字典中的某个单词
train_data_test = [1, 8, 3, 6, 7, 9, 8, 5, 5, 4]
train_labels_test = [0, 1, 1, 0, 1, 1, 1, 0, 1, 0]


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension), dtype="float32")
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    path='D:/python_project/深度学习/keras_t/codes/imdb.npz', num_words=10000)

# word_index = imdb.get_word_index()
# reverse_word_index = dict(
#     [(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data_test])

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# print(x_train[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())
loss_values = history_dict["loss"]  # 训练数据的损失
acc_values = history_dict["acc"]  # 训练数据的准确率
val_loss_values = history_dict["val_loss"]  # 验证数据的损失
val_acc_values = history_dict["val_acc"]   # 验证数据的准确率

# print(f"loss_values: {loss_values}, acc_values: {acc_values}, "
#       f"val_loss_values: {val_loss_values}, val_acc_values: {val_acc_values}")
#
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

# results = model.evaluate(x_test, y_test)
# print(results)
results = model.predict(x_test)
print(results)
