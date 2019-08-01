#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
from PIL import Image
#import torch
#from torch.nn import functional as F
#from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
# def transfer(img, shift_x):
#     img_torch = transforms.ToTensor()(img)#一个把范围取值的英文[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，范围取值的英文[0,1.0]的torch.FloadTensor
#     theta = torch.tensor([
#         [1, 0, shift_x],
#         [0, 1, 0]#对图片进行想右平移shift_x
#
#     ], dtype=torch.float)
#     grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
#     output = F.grid_sample(img_torch.unsqueeze(0), grid)
#     new_img = output[0].numpy().transpose(1, 2, 0)
#     return new_img


def transfer(img, shift_x):
    rows, cols = img.shape
    # 平移矩阵M：[[1,0,x],[0,1,y]]
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    new_img = cv2.warpAffine(img, M, (cols, rows))
    return new_img

ith = 0
list = []
with open('./train_labels.txt', 'r') as f:
    with open('C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/originsizeaug2/train_labels_originsize_transform3.txt', 'a') as result:
        while True:
            line = f.readline()
            if not line:
                break
            # if ith==3:
            #     break
            line = line.split()
            # reader = csv.reader(f)
            # for iter in reader:
            #     if(iter[0]=='file_id'):
            #         continue
            img = cv2.imread('C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/train/'+line[0]+'.png', 0)
            #img = Image.open('C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/trainresize/'+line[0]+'.png')  # 读取图片
            k = np.random.uniform(-1, 1)*86
            img_transfer = transfer(img, k)    # 对图片进行随机平移
            cv2.imwrite('C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/originsizeaug2/originsizetransformaug/'+str(int(line[0])+31500)+'.png',img_transfer)
            # img_transfer = np.uint8(img_transfer*255)
            # out = Image.fromarray(img_transfer).convert('RGB')
            #img.save('C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/trainresizeaug/'+iter[0]+'.png')   # 保存原始图片
            #out.save('C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/trainresizeaug/'+line[0]+'.png')     # 保存随机平移后的图片
            #data = (str(int(line[0])+14500), line[1])   # 随机生成的图片的名称和label保存在list, 后面再写入csv文件
            #list.append(data)
            ith += 1
            result.writelines(str(int(line[0])+31500) + ' ' + line[1] + '\n')
    print('Conversion is over!')
# with open('./train_labels_transform.txt', 'a+') as f:
#     writer = csv.writer(f)
#     for i in range(len(list)):
#         writer.writerow(list[i])




