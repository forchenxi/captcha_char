import keras
from keras.models import load_model
import numpy as np
# from PIL import Image
import cv2
import os
from string import digits, ascii_lowercase

alphanumeric = ascii_lowercase + digits
# model = load_model('C:/Users/admin/Downloads/新建文件夹/my_cnn_ningbo_softmax.h5')
model = load_model('D:/captcha/nhc/cnn_nhc_I.h5')
# model.summary()

x_val = np.load("x_val.npy")
y_val = np.load("y_val.npy")


def convert_labels(y):
    y1, y2, y3, y4 = [], [], [], []
    for capt in y:
        y1.append(capt[0])
        y2.append(capt[1])
        y3.append(capt[2])
        y4.append(capt[3])
    return [y1, y2, y3, y4]


# result = model.evaluate(x_val, convert_labels(y_val))
# print(model.metrics_names)
# print(result)
# img = cv2.imread('D:/captcha/credit_gx_test/2BUY_0.4279397312997719.jpg')
# img = cv2.medianBlur(img, 3)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# x = np.reshape(img, [1, 40, 120, 1])
# x = x.astype('float32')
# y = model.predict(x)
# print(y)
# print(np.array(y).shape)
# print(np.argmax(y, axis=2))
# result = ''.join((alphanumeric[i[0]] for i in np.argmax(y, axis=2)))
# print(result)

_path = 'D:/captcha/nhc/test_new/'
# _path = 'val_imgs/'
i = 0
for f in os.listdir(_path):
    if not f.endswith('jpg'):
        continue
    img = cv2.imread(_path + f)
    # 中值模糊
    img = cv2.medianBlur(img, 3)
    # 转灰度图
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 自适应阈值二值化
    # img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    x = np.reshape(img, [1, 40, 135, 3])
    x = x.astype('float32')
    x /= 255
    y = model.predict(x)
    captcha = f[:4].lower()
    result = ''.join((alphanumeric[i[0]] for i in np.argmax(y, axis=2)))
    print('img {} predict: {}'.format(captcha, result))
    if result == captcha.lower():
        i += 1

total_imgs = len(os.listdir(_path))
print("{}/{}".format(i, total_imgs))
accuracy = i / total_imgs
print('accuracy: {}'.format(accuracy))


# A完全不处理，准确率：0.626
# B使用数据增强（旋转15度和缩放0.1），准确率：0.677
# C使用中值模糊，准确率：0.638
# D使用中值模糊+数据增强（旋转15度和缩放0.1）准确率：0.701
# E使用中值模糊+数据增强（旋转10度和缩放0.05）准确率：0.725

# F去除干扰线+数据增强（旋转15度和缩放0.1）准确率：0.713
# G去除干扰线+中值模糊，准确率：0.698
# H去除干扰线+中值模糊+数据增强（旋转15度和缩放0.1）准确率：0.754
# I去除干扰线+中值模糊+数据增强（旋转10度和缩放0.05）准确率：0.748
