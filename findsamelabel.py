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
#十折交叉验证，找到每次在测试集上都一样的数据（一样则记录下正确标签和图像名字）
label_path="C:/seafile/Seafile/yqy/zhizibei/speech-spectrograms/compare.txt"
count=-1
samelabel=np.zeros(5377)
diff1label=np.zeros(5377)
diff2label=np.zeros(5377)
diff3label=np.zeros(5377)
diff4label=np.zeros(5377)
diff5label=np.zeros(5377)

diff1withdiff2=np.zeros(5377)
imgname=np.zeros(5377)
label0=np.zeros(5377)
label1=np.zeros(5377)
label2=np.zeros(5377)
reallabel=np.zeros(5377)
xiugaizhi=0
countsamelabel=0
for i in range(5377):
    imgname[i]=20000
    #samelabel[i]=2222
with open(label_path) as labels:
    with open('C:/seafile/Seafile/yqy/zhizibei/codeyqy/tenlabel.txt', 'w') as result:
        while True:
            count += 1
            if count ==5377:
                break
            line = labels.readline()
            if not line:
                break
            line = line.split()


            for labelyqy in range(0,4):
                if int(line[labelyqy])==0:
                    label0[count]+=1
                elif int(line[labelyqy])==1:
                    label1[count]+=1
                elif int(line[labelyqy])==2:
                    label2[count]+=1
            reallabel=int(line[2])
            #reallabel=0
            if label0[count]>label1[count]and label0[count]>label2[count]:
                if reallabel!=0:
                    reallabel=0
                    xiugaizhi += 1
            if label1[count]>label0[count]and label1[count]>label2[count]:
                if reallabel != 1:
                    reallabel=1
                    xiugaizhi += 1
            if label2[count]>label0[count] and label2[count]>label1[count]:
                if reallabel != 2:
                    reallabel = 2
                    xiugaizhi += 1
            # if label1[count]>label0[count] or label2[count]>label0[count]:
            #     xiugaizhi+=1
            result.writelines(str(reallabel) + '\n')
        print(xiugaizhi)



            # if line[0]==line[1]==line[2]==line[3]==line[4]==line[5]==line[6]==line[7]==line[8]==line[9]==line[10]==line[11]==line[12]==line[13]==line[14]==line[15]==line[16]==line[17]==line[18]==line[19]==line[20]==line[21]==line[22]==line[23]==line[24]==line[25]==line[26]==line[27]==line[28]==line[29]==line[30]==line[31]==line[32]==line[33]==line[34]==line[35]==line[36]==line[37]==line[38]==line[39]==line[40]:
            # #if line[0] == line[1]:
            #     countsamelabel+=1
            #     imgname[count] +=count
            #     samelabel[count]=line[0]
            #     result.writelines(str(int(imgname[count]))+' '+str(int(samelabel[count])) + '\n')
            # #result.writelines(str(int(samelabel[count])) + '\n')

            # #if not(line[0] == line[1]):
            # if not (line[0] == line[1] == line[2] == line[3] == line[4]):
            #     imgname[count] += count
            #     diff1label[count] = line[0]
            #     diff2label[count] = line[1]
            #     # diff3label[count] = line[2]
            #     # diff4label[count] = line[3]
            #     # diff5label[count] = line[4]
            #     # if int(diff1label[count])==0:
            #     #     if int(diff2label[count])==1:
            #     #         diff1withdiff2[count]=2
            #     #     if int(diff2label[count])==2:
            #     #         diff1withdiff2[count] = 1
            #     # if int(diff1label[count])==1:
            #     #     if int(diff2label[count])==0:
            #     #         diff1withdiff2[count]=2
            #     #     if int(diff2label[count])==2:
            #     #         diff1withdiff2[count] = 0
            #     # if int(diff1label[count])==2:
            #     #     if int(diff2label[count])==0:
            #     #         diff1withdiff2[count]=1
            #     #     if int(diff2label[count])==1:
            #     #         diff1withdiff2[count] = 0
            #     result.writelines(str(int(imgname[count])) + ' ' + str(int(diff1label[count])) + ' ' + str(
            #     int(diff2label[count]))  + '\n')

            # result.writelines(str(int(imgname[count])) + ' ' + str(int(diff1label[count]))+' ' + str(int(diff2label[count]))+' ' + str(int(diff3label[count])) +' ' + str(int(diff4label[count]))+' ' + str(int(diff5label[count])) +'\n')
                #result.writelines( str(int(diff1withdiff2[count])) + '\n')
        #print (countsamelabel)
            # if line[0]!=line[1]:
            #     result.writelines(line[0] + ' ' + line[1] + '\n')
