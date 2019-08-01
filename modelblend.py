#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import caffe
from caffe import layers as L,params as P
import os
from pylab import *
import cv2
caffe.set_device(0)
caffe.set_mode_gpu() #设置为GPU运行
model_def1 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/resnet18_deploy1xiugai2.prototxt'#不是solver！！！          注意文件中的batchsize必须设置为1才能按顺序输出所有的
model_def2 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai.prototxt'
model_def3 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/resnet18_deploy1xiugai3.prototxt'
model_def4 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai4.prototxt'
model_def5 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai5.prototxt'
model_def6 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai6.prototxt'
model_def7 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai7.prototxt'
model_def8 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai8.prototxt'
model_def9 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai9.prototxt'
model_def10 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai10.prototxt'
model_def11 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai11.prototxt'
model_def12 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai12.prototxt'
model_def13 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai13.prototxt'
model_def14 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai14.prototxt'
model_def15 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai15.prototxt'
model_def16 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai16.prototxt'
model_def17 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai17.prototxt'
model_def18 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/formodelblend/resnet18_deploy1xiugai18.prototxt'

model_weights1 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190723origin_add_maskverandhoriandmix_transform1to4/190723_iter_9400.caffemodel ' # 修改成你的caffemodel文件的路径
model_weights2 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190724originsize/190724_iter_5000.caffemodel'
model_weights3 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190723originaddtransformmask/190723_iter_7800.caffemodel'
model_weights4 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190723trainresizetransformaug1and2/190723_iter_5400.caffemodel'
model_weights5 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190718snap1/190718_iter_2500.caffemodel'
model_weights6 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190725originsizeshizhe2qiepian30_2/190725_iter_1600.caffemodel'
model_weights7 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190725originsizetrainandtest/190725_iter_4000.caffemodel'
model_weights8 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190725originsizeshizhe3qiepian50/190725_iter_4200.caffemodel'
model_weights9 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190725originsizeshizhe3qiepian20/190725_iter_8800.caffemodel'
model_weights10 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190725originsizeshizhe3qiepian50_batchsizetest/190725_iter_5400.caffemodel'
model_weights11 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190725originsize_add_mask_transform1to6/190725_iter_4600.caffemodel'
model_weights12 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190726originsizeshizhe4qiepian47/190726_iter_6000.caffemodel'
model_weights13 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190726originsizeshizhe5qiepian40/190726_iter_4200.caffemodel'
model_weights14 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190726origintrainand26sametestqiepian45/190726_iter_4000.caffemodel'
model_weights15 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190727originsizeaug2step/190726_iter_2000.caffemodel'
model_weights16 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190731uniform/190731_iter_5600.caffemodel'
model_weights17 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190731gaussian/190731_iter_3800.caffemodel'
model_weights18 = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190726originsizeaug2/190726_iter_4400.caffemodel'
if __name__=="__main__":
   # net = caffe.Net(model_def, model_weights, caffe.TEST)
    net1 = caffe.Net(model_def1,1, weights=model_weights1)
    net2 = caffe.Net(model_def2, 1, weights=model_weights2)
    net3 = caffe.Net(model_def3, 1, weights=model_weights3)
    net4 = caffe.Net(model_def4, 1, weights=model_weights4)
    net5 = caffe.Net(model_def5, 1, weights=model_weights5)
    net6 = caffe.Net(model_def6, 1, weights=model_weights6)
    net7 = caffe.Net(model_def7, 1, weights=model_weights7)
    net8 = caffe.Net(model_def8, 1, weights=model_weights8)
    net9 = caffe.Net(model_def9, 1, weights=model_weights9)
    net10 = caffe.Net(model_def10, 1, weights=model_weights10)
    net11 = caffe.Net(model_def11, 1, weights=model_weights11)
    net12 = caffe.Net(model_def12, 1, weights=model_weights12)
    net13 = caffe.Net(model_def13, 1, weights=model_weights13)
    net14 = caffe.Net(model_def14, 1, weights=model_weights14)
    net15 = caffe.Net(model_def15, 1, weights=model_weights15)
    net16 = caffe.Net(model_def16, 1, weights=model_weights16)
    net17 = caffe.Net(model_def17, 1, weights=model_weights17)
    net18 = caffe.Net(model_def18, 1, weights=model_weights18)
    test_batch=1
    test_num=5377
    test_N=int(np.ceil(test_num/test_batch))
    with open('C:/seafile/Seafile/yqy/zhizibei/codeyqy/testyqy.txt') as image_list: # 修改成你要测试的txt文件的路径，这个txt文件的内容一般是：每行表示图像的路径，然后空格，然后是标签，也就是说每行都是两列
        with open('C:/seafile/Seafile/yqy/zhizibei/codeyqy/predict.txt','w') as result: # 如果你想把预测的结果写到一个txt文件中，那么把这个路径修改成你想保存这个txt文件的路径
            countyqy=-1
            correctyqy=0
            for i in range(test_N):
                output1 = net1.forward()
                output_prob1 = net1.blobs['fc2_accent'].data[0]
                output2 = net2.forward()
                output_prob2 = net2.blobs['fc2_accent'].data[0]
                output3 = net3.forward()
                output_prob3 = net3.blobs['fc2_accent'].data[0]
                output4 = net4.forward()
                output_prob4 = net4.blobs['fc2_accent'].data[0]
                output5 = net5.forward()
                output_prob5 = net5.blobs['fc2_accent'].data[0]
                output6 = net6.forward()
                output_prob6 = net6.blobs['fc2_accent'].data[0]
                output7 = net7.forward()
                output_prob7 = net7.blobs['fc2_accent'].data[0]
                output8 = net8.forward()
                output_prob8 = net8.blobs['fc2_accent'].data[0]
                output9 = net9.forward()
                output_prob9 = net9.blobs['fc2_accent'].data[0]
                output10 = net10.forward()
                output_prob10 = net10.blobs['fc2_accent'].data[0]
                output11 = net11.forward()
                output_prob11 = net11.blobs['fc2_accent'].data[0]
                output12 = net12.forward()
                output_prob12 = net12.blobs['fc2_accent'].data[0]
                output13 = net13.forward()
                output_prob13 = net13.blobs['fc2_accent'].data[0]
                output14 = net14.forward()
                output_prob14 = net14.blobs['fc2_accent'].data[0]
                output15 = net15.forward()
                output_prob15 = net15.blobs['fc2_accent'].data[0]
                output16 = net16.forward()
                output_prob16 = net16.blobs['fc2_accent'].data[0]
                output17 = net17.forward()
                output_prob17 = net17.blobs['fc2_accent'].data[0]
                output18 = net18.forward()
                output_prob18 = net18.blobs['fc2_accent'].data[0]
                correctyqy += sum((output_prob1).argmax() == net1.blobs['label'].data)
                #correctyqy+=sum(output_prob1.argmax()==net1.blobs['label'].data)
                result.writelines(str((2*output_prob1+output_prob2+output_prob3+output_prob4+output_prob5+output_prob6+output_prob7+output_prob8+output_prob9+output_prob10+output_prob11+output_prob12+output_prob13+output_prob14+output_prob15+output_prob16*5+output_prob17+output_prob18).argmax())+'\n')
                print(correctyqy)