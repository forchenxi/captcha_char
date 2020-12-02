import os
import random
from os import remove


# ss = 'abcdefghijklmnopqrstuvwxyz'
# ss = '0'

# img_path = 'D:/deeplearning/creditGX/imgs/'
# imgs_list = os.listdir(img_path)
# for i in range(1250):
#     t = int(random.random() * len(imgs_list))
#     i_path = imgs_list[t]
#     with open(img_path + i_path, 'rb') as f1:
#         with open('D:/deeplearning/creditGX/test_imgs/{}'.format(i_path), 'wb') as f2:
#             f2.write(f1.read())
#             print('copy {} success'.format(i_path))
#     remove(img_path + i_path)
#     del imgs_list[t]


# img_path = 'd:/learning/ningboGeneral/min/train_imgs/'
# imgs_list = os.listdir(img_path)
# for i in range(2000):
#     t = int(random.random() * len(imgs_list))
#     i_path = imgs_list[t]
#     with open(img_path + i_path, 'rb') as f1:
#         with open('d:/learning/ningboGeneral/min/test_imgs/{}'.format(i_path), 'wb') as f2:
#             f2.write(f1.read())
#             print('copy {} success'.format(i_path))
#     remove(img_path + i_path)
#     del imgs_list[t]


img_path = 'D:/captcha/nhc/train/'
imgs_list = os.listdir(img_path)
for i in range(899):
    t = int(random.random() * len(imgs_list))
    i_path = imgs_list[t]
    with open(img_path + i_path, 'rb') as f1:
        with open('D:/captcha/nhc/test/{}'.format(i_path), 'wb') as f2:
            f2.write(f1.read())
            print('copy {} success'.format(i_path))
    remove(img_path + i_path)
    del imgs_list[t]
