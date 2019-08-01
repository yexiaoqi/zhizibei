#!/usr/bin/env python
# -*- coding: utf-8 -*-
import caffe
import lmdb
import numpy as np
import cv2
from caffe.proto import caffe_pb2
#读取制作的lmdb的data:
def read_lmdb(lmdb_file):
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    count = 0
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)

        label = datum.label
        data = caffe.io.datum_to_array(datum)
        print data.shape
        print datum.channels
        #image = data
        image = data.transpose(1, 2, 0)
        cv2.imshow('cv2.png', image)
        cv2.waitKey(0)
        count = count + 1
        # if count==2:
        #     cv2.imwrite('test.png', image)

        print(label)
        #print('Number of items: {}'.format(count))

        #data = caffe.io.datum_to_array(datum)
        #print('{},{}'.format(key, data[:,0,0]))

def main():
    #lmdb_file = 'E:/face_attribute_recognition_yqy/yht/allarrt/data/train_labels_lmdb'
    lmdb_file = 'C:/seafile/Seafile/yqy/zhizibei/codeyqy/train_labels_originsize_add_mask_transform1to6'
    read_lmdb(lmdb_file)

if __name__ == '__main__':
    main()

