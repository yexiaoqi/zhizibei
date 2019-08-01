#!/usr/bin/env python
# -*- coding: utf-8 -*-

#用训练完的caffemodel计算测试集所有属性的accuracy和年龄的mae
# import caffe
# from caffe import layers as L,params as P
# import numpy as np
# from pylab import  *
# import os
# from xlwt import *
# def main():
#
#     caffe.set_device(0)
#     caffe.set_mode_gpu()
#     solver = caffe.get_solver('./face_attr_recog_solver.prototxt')
#     #solver.net.copy_from(  'E:/face_attribute_recognition_yqy/180828testproto/mobilenet_v2_iter_106000.caffemodel')
#     solver.net.copy_from(
#         './190716_iter_1700.caffemodel ')
#     niter = 20
#     labels = ['0', '1', '2']
#     solver.step(1)
#     for it in range(niter):
#         print 'Iteration',it,'testing'
#         solver.test_nets[0].forward()
#         for batch in range(0, 25):
#             solver.test_nets[0].blobs['fc2_' + labels[attr]].data[batch, 0]



import sys
import caffe
from caffe import layers as L,params as P
import os
from pylab import *
import cv2
caffe.set_device(0)
caffe.set_mode_gpu() #设置为GPU运行


# 修改成你的deploy.prototxt文件路径
# model_def = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/resnet18_deploy1xiugai.prototxt'#不是solver！！！
# model_weights = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190716_iter_2500.caffemodel ' # 修改成你的caffemodel文件的路径


# solver = caffe.get_solver('./face_attr_recog_solver.prototxt')
# solver.net.copy_from(
#         './190716_iter_2500.caffemodel ')
#solver.test_nets[0].forward()
#solver.step(1)
#这是一个由mean.binaryproto文件生成mean.npy文件的函数
# def convert_mean(binMean,npyMean):
#     blob = caffe.proto.caffe_pb2.BlobProto()
#     bin_mean = open(binMean, 'rb' ).read()
#     blob.ParseFromString(bin_mean)
#     arr = np.array( caffe.io.blobproto_to_array(blob) )
#     npy_mean = arr[0]
#     np.save(npyMean, npy_mean )
# binMean='C:/seafile/Seafile/yqy/zhizibei/codeyqy/train_mean_190715.binaryproto' #修改成你的mean.binaryproto文件的路径
# npyMean='C:/seafile/Seafile/yqy/zhizibei/codeyqy/mean.npy' #你想把生成的mean.npy文件放在哪个路径下
# convert_mean(binMean,npyMean)
#
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))  # 通道变换，例如从(530,800,3) 变成 (3,530,800)
# transformer.set_mean('data', np.load(npyMean).mean(1).mean(1)) #如果你在训练模型的时候没有对输入做mean操作，那么这边也不需要
# transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
# #transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR











# if __name__=="__main__":
#     net = caffe.Net(model_def, model_weights, caffe.TEST)
#
#     transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#     transformer.set_transpose('data', (2, 0, 1))
#     mean_filename = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/train_mean_190715.binaryproto'
#     bin_mean = open(mean_filename, 'rb').read()
#     blob = caffe.io.caffe_pb2.BlobProto()
#     blob.ParseFromString(bin_mean)
#     mean = caffe.io.blobproto_to_array(blob)[0].mean(1).mean(1)
#
#     # transformer.set_mean('data', np.load('mean.npy').mean(1).mean(1))
#
#     transformer.set_mean('data', mean)
#     transformer.set_raw_scale('data', 255)
#     transformer.set_channel_swap('data', (2, 1, 0))
#
#     # # net=caffe.Net(model_def,model_weights,caffe.TEST)
#     # mean_filename = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/train_mean_190715.binaryproto'
#     # bin_mean = open(mean_filename, 'rb').read()
#     # blob = caffe.io.caffe_pb2.BlobProto()
#     # blob.ParseFromString(bin_mean)
#     # mean = caffe.io.blobproto_to_array(blob)[0].mean(1).mean(1)
#     # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#     # transformer.set_transpose('data', (2, 0, 1))
#     # transformer.set_mean('data', mean)
#     # # transformer.set_raw_scale('data',0.00392156862745)
#     # # transformer.set_channel_swap('data',(2,1,0))
#     # net.blobs['data'].reshape(1, 3, 224, 224)
#
#     with open('C:/seafile/Seafile/yqy/zhizibei/codeyqy/testyqy.txt') as image_list: # 修改成你要测试的txt文件的路径，这个txt文件的内容一般是：每行表示图像的路径，然后空格，然后是标签，也就是说每行都是两列
#         with open('C:/seafile/Seafile/yqy/zhizibei/codeyqy/predict.txt','w') as result: # 如果你想把预测的结果写到一个txt文件中，那么把这个路径修改成你想保存这个txt文件的路径
#             count_right=0
#             count_all=0
#             countyqy=0
#             while 1:
#                 countyqy+=1
#                 if countyqy == 300:
#                     break
#                 list_name=image_list.readline()
#                 list_name=list_name.split()
#                 if list_name == '\n' or list_name == '': #如果txt文件都读完了则跳出循环
#                     break
#                 # image_type=list_name[0:-3].split('.')[-1]
#                 # if image_type == 'gif': #这里我对gif个数的图像直接跳过
#                 #     continue
#                 #image = cv2.imread('C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/trainresize/'+str(list_name[0])+'.png')
#                 image = caffe.io.load_image('C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/trainresize/'+str(list_name[0])+'.png')
#                     #image = caffe.io.load_image( 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/test.png')
#                 # 这里要添加你的图像所在的路径，根据你的list_name灵活调整，总之就是图像路径
#                 #imshow(image)
#                 transformed_image = transformer.preprocess('data', image)
#                 #tranformed_image = transformed_image * 0.00392156862745
#                 # 用转换后的图像代替net.blob中的data
#                 net.blobs['data'].data[...] = transformed_image
#                 #net.blobs['data'].reshape(1, 3, 224, 224)
#                 ### perform classification
#                 output = net.forward()
#             # 读取预测结果和真实label
#                 #output_prob = output['fc2_accent'].data[0]
#                 output_prob = net.blobs['fc2_accent'].data[0]
#                 true_label = int(list_name[1])
#         # 如果预测结果和真实label一样，则count_right+1
#                 if(output_prob.argmax()==true_label):
#                     count_right=count_right+1
#                 count_all=count_all+1
#
#         # 保存预测结果，这个可选
#                 result.writelines(list_name[1]+' '+str(output_prob.argmax())+'\n')
#         #可以每预测完100个样本就打印一些，这样好知道预测的进度，尤其是要预测几万或更多样本的时候，否则你还以为代码卡死了
#                 if(count_all%100==0):
#                     print count_all
#
#            # 打印总的预测结果
#             print 'Accuracy: '+ str(float(count_right)/float(count_all))
#             print 'count_all: ' + str(count_all)
#             print 'count_right: ' + str(count_right)
#             print 'count_wrong: ' + str(count_all-count_right)
#
#


# #以下使用修改的deploy文件，归一化减均值等操作均放在deploy文件中，而不在python代码里写，放在python代码里好像因为是灰度图像会出现问题
# model_def = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/resnet18_deploy1xiugai.prototxt'#不是solver！！！
# model_weights = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190717snap1/190716_iter_500.caffemodel ' # 修改成你的caffemodel文件的路径
# if __name__=="__main__":
#     net = caffe.Net(model_def, model_weights, caffe.TEST)
#     with open('C:/seafile/Seafile/yqy/zhizibei/codeyqy/testyqy.txt') as image_list: # 修改成你要测试的txt文件的路径，这个txt文件的内容一般是：每行表示图像的路径，然后空格，然后是标签，也就是说每行都是两列
#         with open('C:/seafile/Seafile/yqy/zhizibei/codeyqy/predict.txt','w') as result: # 如果你想把预测的结果写到一个txt文件中，那么把这个路径修改成你想保存这个txt文件的路径
#             count_right=0
#             count_all=0
#             countyqy=-1
#             while 1:
#                 countyqy+=1
#                 if countyqy == 300:
#                     break
#                 list_name=image_list.readline()
#                 list_name=list_name.split()
#                 if countyqy%10==0:
#                     continue
#
#                 if list_name == '\n' or list_name == '': #如果txt文件都读完了则跳出循环
#                     break
#                 output = net.forward()
#             # 读取预测结果和真实label
#                 #output_prob = output['fc2_accent'].data[0]
#                 output_prob = net.blobs['fc2_accent'].data[0]
#                 true_label = int(list_name[1])
#         # 如果预测结果和真实label一样，则count_right+1
#                 if(output_prob.argmax()==true_label):
#                     count_right=count_right+1
#                 count_all=count_all+1
#
#         # 保存预测结果，这个可选
#                 result.writelines(list_name[1]+' '+str(output_prob.argmax())+'\n')
#         #可以每预测完100个样本就打印一些，这样好知道预测的进度，尤其是要预测几万或更多样本的时候，否则你还以为代码卡死了
#                 if(count_all%100==0):
#                     print count_all
#
#            # 打印总的预测结果
#             print 'Accuracy: '+ str(float(count_right)/float(count_all))
#             print 'count_all: ' + str(count_all)
#             print 'count_right: ' + str(count_right)
#             print 'count_wrong: ' + str(count_all-count_right)





#以下使用修改的deploy文件，归一化减均值等操作均放在deploy文件中，而不在python代码里写，放在python代码里好像因为是灰度图像会出现问题
model_def = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/resnet18_deploy1xiugai.prototxt'#不是solver！！！          注意文件中的batchsize必须设置为1才能按顺序输出所有的
model_weights = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/190726originsizeaug2/190726_iter_3400.caffemodel ' # 修改成你的caffemodel文件的路径
if __name__=="__main__":
   # net = caffe.Net(model_def, model_weights, caffe.TEST)
    net = caffe.Net(model_def,1, weights=model_weights)
    test_batch=1
    test_num=450
    test_N=int(np.ceil(test_num/test_batch))
    with open('C:/seafile/Seafile/yqy/zhizibei/codeyqy/testyqy.txt') as image_list: # 修改成你要测试的txt文件的路径，这个txt文件的内容一般是：每行表示图像的路径，然后空格，然后是标签，也就是说每行都是两列
        with open('C:/seafile/Seafile/yqy/zhizibei/codeyqy/predict.txt','w') as result: # 如果你想把预测的结果写到一个txt文件中，那么把这个路径修改成你想保存这个txt文件的路径
            # count_right=0
            # count_all=0
            countyqy=-1
            correctyqy=0
            for i in range(test_N):
            # while 1:
                #countyqy+=1
                # if countyqy == 300:
                #     break
                # list_name=image_list.readline()
                # list_name=list_name.split()
                # if countyqy%10==0:
                #     continue

                # if list_name == '\n' or list_name == '': #如果txt文件都读完了则跳出循环
                #     break
                output = net.forward()
            # 读取预测结果和真实label
                #output_prob = output['fc2_accent'].data[0]
                output_prob = net.blobs['fc2_accent'].data[0]
                correctyqy+=sum(output_prob.argmax()==net.blobs['label'].data)
               # true_label = int(list_name[1])
        # 如果预测结果和真实label一样，则count_right+1
        #         if(output_prob.argmax()==true_label):
        #             count_right=count_right+1
        #         count_all=count_all+1

        # 保存预测结果，这个可选
                result.writelines(str(output_prob.argmax())+'\n')
                print(correctyqy)
        #可以每预测完100个样本就打印一些，这样好知道预测的进度，尤其是要预测几万或更多样本的时候，否则你还以为代码卡死了
           #      if(count_all%100==0):
           #          print count_all
           #
           # # 打印总的预测结果
           #  print 'Accuracy: '+ str(float(count_right)/float(count_all))
           #  print 'count_all: ' + str(count_all)
           #  print 'count_right: ' + str(count_right)
           #  print 'count_wrong: ' + str(count_all-count_right)






