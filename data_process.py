# _*_ coding:utf-8 _*_
"""
@time :2020/3/20 17:49
@desc: 验证码预处理
"""
import cv2
import numpy as np
import os
from string import ascii_lowercase, digits
from PIL import Image


# 文件路径相关参数
train_path = 'D:/captcha/nhc/train_new/'
# val_path = 'D:/captcha/credit_gx_val/'
train_list = os.listdir(train_path)  # 把该路径下的所有图片全部放到train_list中
# val_list = os.listdir(val_path)
train_num = len(train_list)   # 获取训练集的图片数量
# val_num = len(val_list)

# 图像参数
img_h = 40
img_w = 135
channels = 3
labels_len = 4   # 验证码位数
num_classes = 36   # 验证码包含的字符集数量（这里是所有数字和字母）
alphanumeric = ascii_lowercase + digits


# 最终生成的训练集、验证集以及标签
# train_imgs = list()
# val_imgs = list()
# train_labels = list()
# val_labels = list()


def process_img(img_list: list, img_path: str):
    imgs_temp = list()
    labels_temp = list()
    for img in img_list:
        path = f'{img_path}{img}'
        # path = 'D:/captcha/nhc/train_new/1rf20.9282258601949195.jpg'
        # 处理标签值（也就是验证码字符）
        label = img[:4].lower()
        label = process_label(label)

        labels_temp.append(label)

        img = cv2.imread(path)
        # 中值模糊
        img = cv2.medianBlur(img, 3)
        # cv2.imwrite('D:/captcha/nhc/1a6z_3.jpg', img)
        # 均值模糊
        # img = cv2.blur(img, (2, 2))
        # 高斯模糊
        # img = cv2.GaussianBlur(img, (5, 5), 1)
        # cv2.namedWindow('captcha', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow('captcha', 40, 135)
        # cv2.imshow('captcha', img)
        # cv2.waitKey(0)

        # 转灰度图
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('captcha', img)
        # cv2.waitKey(0)
        # Otsu's 二值化
        # ret, img = cv2.threshold(img, 0, 1, cv2.THRESH_OTSU)
        # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        # img /= 255
        # 自适应阈值二值化
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # cv2.imshow('captcha', img)
        # cv2.waitKey(0)
        img = np.array(img, dtype='float32')
        img /= 255
        imgs_temp.append(np.reshape(img, (img_h, img_w, channels)))
    # imgs_temp = np.array(imgs_temp, dtype='float32')
    labels_temp = np.array(labels_temp, dtype='float32')
    return imgs_temp, labels_temp


def process_label(label):
    result = np.zeros((labels_len, num_classes), dtype='float32')
    x = label.lower()
    for i, c in enumerate(x):
        result[i][alphanumeric.index(c)] = 1

    return result


# 将图片从P模式转为RGB模式，使cv2可以打开
def convert_image(img_list: list, img_path: str):
    for img in img_list:
        path = f'{img_path}{img}'
        img_s = Image.open(path)
        img_s = img_s.convert('RGB')
        img_s.save(f'D:/captcha/nhc/train/{img}')


# convert_image(train_list, train_path)
# 训练数据处理
train_imgs, train_labels = process_img(train_list, train_path)
np.save('D:/captcha/nhc/x_train', train_imgs)
np.save('D:/captcha/nhc/y_train', train_labels)

# 验证数据处理
# val_imgs, val_labels = process_img(val_list, val_path)
# np.save('x_val', val_imgs)
# np.save('y_val', val_labels)
