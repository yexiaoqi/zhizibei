#!/usr/bin/env python
# -*- coding: utf-8 -*-
import caffe
import cv2
for imgcount in range(10000,14500):
    pic = cv2.imread('C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/train/' + str(imgcount) + '.png', cv2.IMREAD_UNCHANGED)
    pic = cv2.resize(pic, (256, 128), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/trainresize2/"+str(imgcount)+".png",pic)
#
#
# for imgcount in range(1,21):
#     pic = cv2.imread('C:\\vsprojects\\test\\test\\result3\\rgb\\' + str(imgcount) + '.png', cv2.IMREAD_UNCHANGED)
#     pic = cv2.resize(pic, (640, 480), interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite("C:\\vsprojects\\test\\test\\xiugairesult2\\rgbresize\\"+str(imgcount)+".png",pic)



# for imgcount in range(1,21):
#     pic = cv2.imread('C:\\vsprojects\\test\\test\\result2\\depth\\' + str(imgcount) + '.png', cv2.IMREAD_UNCHANGED)
#     pic = cv2.resize(pic, (640, 480), interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite("C:\\vsprojects\\test\\test\\xiugairesult2\\withbackgraounddepthresize\\"+str(imgcount)+".png",pic)


# pic = cv2.imread('C:/vsprojects/Robust-Color-Guided-Depth-Map-Restoration-master/Robust-Color-Guided-Depth-Map-Restoration-master/yqy/2inpaintopencv.png', cv2.IMREAD_UNCHANGED)
# # pic = cv2.resize(pic, (960,540), interpolation=cv2.INTER_CUBIC)#这样出来的结果是错的，有多个平面
# pic = cv2.resize(pic, (960,540))
# #print pic.shape
# cv2.imwrite("C:/vsprojects/Robust-Color-Guided-Depth-Map-Restoration-master/Robust-Color-Guided-Depth-Map-Restoration-master/yqy/22inpaintopencvdownsize.png",pic)