# _*_ coding:utf-8 _*_
"""
@time :2020/6/5 12:47
@author :liutengfei
@mail:liutengfei@bertadata.com
@desc: 去除验证码图片的干扰线
"""
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# 文件路径相关参数
train_path = 'D:/captcha/nhc/train/'
train_list = os.listdir(train_path)


def process_img(img_list: list, img_path: str):
    for img in img_list:
        path = f'{img_path}{img}'
        image = Image.open(path)
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                r, g, b = image.getpixel((i, j))
                if r > 110 or g > 100:
                    image.putpixel((i, j), (255, 255, 255))

        image.save(f'D:/captcha/nhc/train_new/{img}')


# 绿色 [51, 255, 51]
# 嫩绿色 [153, 213, 51]
# 深绿色 [51, 85, 51]
# 黄色 [255, 255, 102]
# 棕色 [102, 43, 51]
# 粉红色（较亮） [255, 0, 153]
# 粉红色（较暗） [204, 85, 102]
# 橙色 [204, 85, 0]
# [51, 85, 204]

# 验证码的颜色
# 深蓝色 [0, 0, 255]
# [51, 0, 204]
# [102, 43, 102]
# [102, 0, 102]
# [102, 0, 51]
# process_img(train_list, train_path)
with open('D:/captcha/nhc/train/1awy0.8460237480620968.jpg', 'rb') as f:
    image_content = f.read()

# PIL Image直接读取二进制流
# img = Image.open(BytesIO(image_content))
# img.show()

# 将Image对象转换为二进制流
# out = BytesIO()
# img.save(out, format("JPEG"))
# image_content = out.getvalue()

# OpenCV cv2直接读取二进制流
img = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
# cv2.imshow('captcha', img)
# cv2.waitKey(0)

# 将cv2对象转换为二进制流
res, out = cv2.imencode('.jpg', img)
image_content = out.tobytes()

img = Image.open(BytesIO(image_content))
img.show()
