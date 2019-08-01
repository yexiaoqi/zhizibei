#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import glob
import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 173
IMAGE_HEIGHT = 128

# train_lmdb = "C:/seafile/Seafile/yqy/zhizibei/codeyqy/190715train_img_lmdbcounttest"
# validation_lmdb ="C:/seafile/Seafile/yqy/zhizibei/codeyqy/190715val_img_lmdbcounttest"
# test_lmdb ="C:/seafile/Seafile/yqy/zhizibei/codeyqy/190715test_img_lmdb"
# #image_path = "C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/trainresize/"
# image_path = "C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/trainresize/"
# label_path="C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/train_labels.txt"



train_lmdb = "C:/seafile/Seafile/yqy/zhizibei/codeyqy/trainandtest190730_190715train_img_lmdb"
validation_lmdb ="C:/seafile/Seafile/yqy/zhizibei/codeyqy/trainandtest190730_190715val_img_lmdb"
test_lmdb ="C:/seafile/Seafile/yqy/zhizibei/codeyqy/size256128_190715test_img_lmdb"
#image_path = "C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/trainresize/"
image_path = "C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/origintrainandtest/"
label_path="C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/trainandtest190730shuf.txt"

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Histogram Equalization
#    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
#    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
#    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

#    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        #data = img.tobytes() )
        #data = img.tobytes() )
        data=np.rollaxis(img, 2).tobytes())

count=-1
key=0
countlabel0=0
countlabel1=0
countlabel2=0
counttrain=0
print '\nCreating train_lmdb'
in_db = lmdb.open(validation_lmdb, map_size=int(4e8))#20万3.5e10
in_txn=in_db.begin(write=True)
with open(label_path) as labels:
    while True:
        count += 1
        # if count ==8876:
        #     break
        line = labels.readline()
        if not line:
            break
        line = line.split()
        # if int(line[1])==0:
        #     counttrain += 1
        # if int(line[1])==1:
        #     countlabel1+=1
        #     if countlabel1 >= 1080:
        #         continue
        #     else:
        #         counttrain+=1
        # if int(line[1])==2:
        #     countlabel2 += 1
        #     if countlabel2 >= 1488:
        #         continue
        #     else:
        #         counttrain+=1


        # if int(line[1])==0:
        #     continue
        # if int(line[1])==1:
        #     countlabel1+=1
        #     if countlabel1 <= 1080:
        #         continue
        #     else:
        #         counttrain+=1
        # if int(line[1])==2:
        #     countlabel2 += 1
        #     if countlabel2 <= 1488:
        #         continue
        #     else:
        #         counttrain+=1



        # if count % 10!= 3:
        #     if int(line[1]) == 0:
        #         countlabel0 += 1
        #     elif int(line[1]) == 1:
        #         countlabel1 += 1
        #     elif int(line[1]) ==2:
        #         countlabel2 += 1
        #     continue



        if int(line[0]) % 10!= 8:
            continue


        # if (int(line[0])>=0 and int(line[0])<=4049) or (int(line[0])>=10000 and int(line[0])<=14049) or (int(line[0])>=14500 and int(line[0])<=18549) or (int(line[0])>=23500 and int(line[0])<=27549):
        #     continue
        # if (int(line[0])>=8100 and int(line[0])<=8549) or (int(line[0])>=14050 and int(line[0])<=14499) or (int(line[0])>=18550 and int(line[0])<=18999) or (int(line[0])>=27550 and int(line[0])<=27999):
        #     continue


        # if (int(line[0]) >= 10000 and int(line[0]) <= 14049) or (int(line[0]) >= 14500 and int(line[0]) <= 18549) or (int(line[0]) >= 32550 and int(line[0]) <= 36549) :
        #     continue
        # if (int(line[0]) >= 14050 and int(line[0]) <= 14499) or (int(line[0]) >= 18550 and int(line[0]) <= 18999) or (int(line[0]) >= 36550 and int(line[0]) <= 36999) :
        #     continue

        # if (int(line[0]) >= 10000 and int(line[0]) <= 14049) or (int(line[0]) >= 23500 and int(line[0]) <= 27549) or (int(line[0]) >= 28000 and int(line[0]) <= 32049) :
        #     continue
        # if (int(line[0]) >= 27550 and int(line[0]) <= 27999) or (int(line[0]) >= 14050 and int(line[0]) <= 14499) or (int(line[0]) >= 32050 and int(line[0]) <= 32499) :
        #     continue

        # if (int(line[0])>=0 and int(line[0])<=4049) or (int(line[0])>=10000 and int(line[0])<=14049) or (int(line[0])>=14500 and int(line[0])<=18549) or (int(line[0])>=23500 and int(line[0])<=27549)or (int(line[0]) >= 28000 and int(line[0]) <= 32049)or (int(line[0]) >= 32550 and int(line[0]) <= 36549)or (int(line[0]) >= 37000 and int(line[0]) <= 41049)or (int(line[0]) >= 41500 and int(line[0]) <= 455449)or (int(line[0]) >= 46000 and int(line[0]) <= 50049):
        #     continue
        # if (int(line[0])>=8100 and int(line[0])<=8549) or (int(line[0])>=14050 and int(line[0])<=14499) or (int(line[0])>=18550 and int(line[0])<=18999) or (int(line[0])>=27550 and int(line[0])<=27999)or (int(line[0]) >= 32050 and int(line[0]) <= 32499)or(int(line[0]) >= 36550 and int(line[0]) <= 36999)or(int(line[0]) >= 41050 and int(line[0]) <= 41499)or(int(line[0]) >= 45550 and int(line[0]) <= 45999)or(int(line[0]) >= 50050 and int(line[0]) <= 50499):
        #     continue

        # if (int(line[0])>=0 and int(line[0])<=4049) or (int(line[0])>=10000 and int(line[0])<=14049) or (int(line[0])>=14500 and int(line[0])<=18549) or (int(line[0])>=23500 and int(line[0])<=27549)or (int(line[0]) >= 28000 and int(line[0]) <= 32049)or (int(line[0]) >= 32550 and int(line[0]) <= 36549)or (int(line[0]) >= 37000 and int(line[0]) <= 41049)or (int(line[0]) >= 41500 and int(line[0]) <= 45549)or (int(line[0]) >= 46000 and int(line[0]) <= 50049)or (int(line[0]) >= 50500 and int(line[0]) <= 54549)or (int(line[0]) >= 55000 and int(line[0]) <= 59049)or(int(line[0]) >= 59500 and int(line[0]) <= 63549):
        #     continue
        # if (int(line[0])>=8100 and int(line[0])<=8549) or (int(line[0])>=14050 and int(line[0])<=14499) or (int(line[0])>=18550 and int(line[0])<=18999) or (int(line[0])>=27550 and int(line[0])<=27999)or (int(line[0]) >= 32050 and int(line[0]) <= 32499)or(int(line[0]) >= 36550 and int(line[0]) <= 36999)or(int(line[0]) >= 41050 and int(line[0]) <= 41499)or(int(line[0]) >= 45550 and int(line[0]) <= 45999)or(int(line[0]) >= 50050 and int(line[0]) <= 50499)or (int(line[0]) >= 54550 and int(line[0]) <= 54999)or (int(line[0]) >= 59050 and int(line[0]) <= 59499)or(int(line[0]) >= 63550 and int(line[0]) <= 63999):
        #     continue




        img = cv2.imread(image_path+line[0]+'.png')
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        #img=cv2.split(img)[0]#只取一个通道，因为是灰度图   不嫩只取一个通道，resnet要求有三个通道
        accent = line[1]
        accent=int(accent)
        datum = make_datum(img, accent)
        in_txn.put('{:0>8d}'.format(key), datum.SerializeToString())
        key+=1
        print '{:0>8d}'.format(key)
        if key % 1000 == 0:
            in_txn.commit()
            in_txn = in_db.begin(write=True)
    print "counttrain"
    print (countlabel0,countlabel1,countlabel2)

# in_db = lmdb.open(test_lmdb, map_size=int(1e9))#20万3.5e10
# in_txn=in_db.begin(write=True)
# with open(label_path) as labels:
#     while True:
#         count += 1
#         if count ==5377:
#             break
#         #line = labels.readline()
#         # if not line:
#         #     break
#         # if count % 10== 0:
#         #     continue
#         #line = line.split()
#         img = cv2.imread(image_path+str(count+20000)+'.png')
#         img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
#         #img=cv2.split(img)[0]#只取一个通道，因为是灰度图   不嫩只取一个通道，resnet要求有三个通道
#         accent = 0
#         accent=int(accent)
#         datum = make_datum(img, accent)
#         in_txn.put('{:0>8d}'.format(key), datum.SerializeToString())
#         key+=1
#         print '{:0>8d}'.format(key)
#         if key % 1000 == 0:
#             in_txn.commit()
#             in_txn = in_db.begin(write=True)

in_txn.commit()
in_db.close()
print '\nFinished processing all images'