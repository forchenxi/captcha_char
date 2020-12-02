# _*_ coding:utf-8 _*_
"""
@time :2020/4/13 19:29
"""

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Conv2D, Activation, MaxPooling2D, Dropout, Dense, Concatenate
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pylab as plt


x_train = np.load("D:/captcha/nhc/x_train.npy")
y_train = np.load("D:/captcha/nhc/y_train.npy")
x_val = np.load("x_val.npy")
y_val = np.load("y_val.npy")

# y_train = y_train.reshape((4, 3000, 36))
# y_val = y_val.reshape((4, 1000, 36))


def convert_labels(y):
    y1, y2, y3, y4 = [], [], [], []
    for capt in y:
        y1.append(capt[0])
        y2.append(capt[1])
        y3.append(capt[2])
        y4.append(capt[3])
    return [y1, y2, y3, y4]


# Data parameters
num_classes = 36
img_shape = (40, 135, 3)
train_nums = 2010
val_nums = 1000

# Network parameters
batch_size = 32
epochs = 50


def data_generator(p_x, p_y, p_batch_size=64):
    x_y_gen = train_datagen.flow(p_x, p_y, batch_size=p_batch_size, shuffle=True)
    while True:
        x, y = x_y_gen.next()
        y1, y2, y3, y4 = (np.array(l) for l in zip(*y))
        yield x, [y1, y2, y3, y4]


def val_generator1(p_batch_size=64):
    x_y_gen = val_datagen.flow(x_val, y_val, batch_size=p_batch_size, shuffle=True)
    while True:
        x, y = x_y_gen.next()
        y1, y2, y3, y4 = (np.array(l) for l in zip(*y))
        yield x, [y1, y2, y3, y4]


# 创建CNN模型
main_input = Input(shape=img_shape, name="inputs")
# 32个卷积核，卷积核大小3*3
conv1 = Conv2D(32, (3, 3), name="conv1", padding="same")(main_input)
# 激活函数优先使用relu
relu1 = Activation('relu', name="relu1")(conv1)
# 最大池化
pool1 = MaxPooling2D(pool_size=(2, 2), padding="same", name="pool1")(relu1)

conv2 = Conv2D(32, (3, 3), name="conv2", padding="same")(pool1)
relu2 = Activation('relu', name="relu2")(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), padding='same', name='pool2')(relu2)

# conv3 = Conv2D(64, (3, 3), name='conv3', padding="same")(pool2)
# relu3 = Activation('relu', name='relu3')(conv3)
# pool3 = MaxPooling2D(pool_size=(2, 2), padding='same', name='pool3')(relu3)

# 全连接层
x = Flatten(name="flatten")(pool2)

x = Dense(256, activation="relu", name="dense1")(x)
# dense2_size = 256
# do1 = Dense(dense2_size, activation='relu', name='dense2_I')(x)
# do2 = Dense(dense2_size, activation='relu', name='dense2_II')(x)
# do3 = Dense(dense2_size, activation='relu', name='dense2_III')(x)
# do4 = Dense(dense2_size, activation='relu', name='dense2_IV')(x)

# out_put1 = Dense(num_classes, activation='softmax', name='out1')(x)
# out_put2 = Dense(num_classes, activation='softmax', name='out2')(x)
# out_put3 = Dense(num_classes, activation='softmax', name='out3')(x)
# out_put4 = Dense(num_classes, activation='softmax', name='out4')(x)

out_put = [Dense(num_classes, activation='softmax', name='out%d' % (i+1))(x) for i in range(1, 5)]
# 创建4个全连接层，区分36类，分别识别4个字符
# output = [Dense(num_classes, activation='softmax', name='out%d' % (i+1))(x) for i in range(4)]

# 输出层,将生成的4个字符拼接输出
# outs = Concatenate()(x)

# 定义模型的输入和输出
model = Model(inputs=main_input, outputs=out_put)
# 查看模型
model.summary()
opt = 'rmsprop'
# opt = keras.optimizers.Adam(lr=0.001)
loss = 'categorical_crossentropy'
model.compile(optimizer=opt, loss=loss, loss_weights=[1., 1., 1., 1.], metrics=['acc'])

early_stopping = EarlyStopping(monitor='loss', patience=10)


train_datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.05)
val_datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.05)

# train_datagen = ImageDataGenerator()
# val_datagen = ImageDataGenerator()

# np.random.seed(200)
# np.random.shuffle(x_train)
# np.random.seed(200)
# np.random.shuffle(y_train)

# 开始训练
# history = model.fit(
#     x_train,
#     convert_labels(y_train),
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(x_val, convert_labels(y_val)),
#     callbacks=[early_stopping]
# )


history = model.fit_generator(
    data_generator(x_train, y_train, batch_size),
    steps_per_epoch=train_nums // batch_size,
    # validation_data=data_generator(x_val, y_val, batch_size),
    # validation_steps=val_nums // batch_size,
    epochs=epochs,
    callbacks=[early_stopping])

model_path = 'D:/captcha/nhc/cnn_nhc_I.h5'
model.save(model_path)

history_dict = history.history
print(history_dict.keys())
loss_values = history_dict["loss"]  # 训练数据的损失(总损失)
acc_values = history_dict["out2_acc"]  # 训练数据的准确率
# val_loss_values = history_dict["val_loss"]  # 验证数据的损失(总损失)
# val_acc_values = history_dict["val_out2_acc"]   # 验证数据的准确率

# 绘制训练损失和验证损失(第一个和第二个参数的维度要一致)
plt.plot(range(1, epochs+1), loss_values, 'bo', label="Training loss")
# plt.plot(range(1, epochs+1), val_loss_values, 'b', label="Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 清空图像
plt.clf()
# 绘制训练精度和验证精度
plt.plot(range(1, epochs+1), acc_values, 'bo', label="Training acc")
# plt.plot(range(1, epochs+1), val_acc_values, 'b', label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
