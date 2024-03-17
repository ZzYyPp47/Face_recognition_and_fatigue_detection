# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2023/7/12 9:49
@version: 1.0
@File: facecnn.py
'''


import os
import cv2
import tensorflow as tf
from tensorflow.keras import datasets,layers,models
from sklearn.model_selection import train_test_split
import random
import numpy as np
from keras.utils import to_categorical

class Data(object):
    def __init__(self,imgsize=64,img_path='./face_list/'):
        self.imgsize=imgsize
        self.img_path=img_path
        # 处理后图片
        self.img_processed=[]
        # 图片标签(原始)
        self.img_labels=[]
        # 训练集
        self.train_img=[]
        self.train_labels=[]
        # 测试集
        self.test_img=[]
        self.test_labels=[]
        # 输入神经网络的形状
        self.input_shape=[]
        self.len_uni_labels=0
        # 标签字典
        self.label_dic=[]
        # 反标签字典
        self.re_label_dic=[]
        # 计算种类并编码
        self.one_hot(img_path)
        # 读取并处理数据
        self.read_path()
        # 分离训练集和测试集
        self.split()

    def process_img(self,img,size_h=None,size_w=None):
        #初始化
        if size_h is None:
            size_h=self.imgsize
        if size_w is None:
            size_w=self.imgsize
        top,bottom,left,right=(0,0,0,0)
        BLACK=[0,0,0]
        #获取图像尺寸
        h,w,_=img.shape
        #对于长宽不等的图片，找到最长的一边
        longest_edge=max(h,w)
        #计算短边需要增加多少像素宽度使其与长边等长
        if h < longest_edge:
            dh=longest_edge-h
            top=dh//2
            bottom=dh-top
        elif w < longest_edge:
            dw=longest_edge-w
            left=dw//2
            right=dw-left
        #填充
        img_add=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)
        #灰度图
        img_add_gray=cv2.cvtColor(img_add,cv2.COLOR_BGR2GRAY)
        #调节大小并返回
        return cv2.resize(img_add_gray,(size_h,size_w))

    def read_path(self,img_path=None):
        #初始化
        if img_path is None:
            img_path=self.img_path
        for dir_item in os.listdir(img_path):
            # 从初始路径开始叠加
            full_path=os.path.abspath(os.path.join(img_path,dir_item))
            if os.path.isdir(full_path):#如果是文件夹，继续递归调用
                self.read_path(full_path)
            else:#文件
                if dir_item.endswith('.png'):
                    img=cv2.imread(full_path)
                    img_processed=self.process_img(img)
                    self.img_processed.append(img_processed)
                    self.img_labels.append(os.path.basename(img_path))  # 只保留文件夹名称作为label

    def one_hot(self,folder_path):
        labels_list = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        self.len_uni_labels=len(labels_list)
        values=range(self.len_uni_labels)
        self.label_dic=dict(zip(labels_list,values))
        self.re_label_dic=dict(zip(values,labels_list))
        print(self.label_dic)


    def split(self, img_rows=None, img_cols=None, img_channels=1, classes=None):
        # 初始化
        if img_rows is None:
            img_rows = self.imgsize
        if img_cols is None:
            img_cols = self.imgsize
        if classes is None:
            classes = self.len_uni_labels
        # 将总数据按0.3的比重随机分配给训练集和测试集
        train_images, test_images, train_labels, test_labels = train_test_split(self.img_processed,self.img_labels, test_size=0.3,random_state=random.randint(0, 100))
        # 将其归一化，图像的各像素值归一化到0~1区间
        train_images = [img / 255.0 for img in train_images]
        test_images = [img / 255.0 for img in test_images]
        # 由于TensorFlow需要通道数，我们添加通道数
        train_images = [img.reshape(img_rows, img_cols, img_channels) for img in train_images]
        test_images = [img.reshape(img_rows, img_cols, img_channels) for img in test_images]
        # 输出训练集、测试集的数量
        #print('we have',len(train_images),'train samples')
        #print('we have',len(test_images),'test samples')
        # 最后转换为np数组
        train_images = np.array(train_images).reshape(len(train_images),img_rows,img_cols,1)
        test_images = np.array(test_images).reshape(len(test_images),img_rows,img_cols,1)
        # 输入神经网络的形状
        self.input_shape = (img_rows, img_cols, img_channels)
        #print('input_shape is',self.input_shape)
        #编码
        train_labels = [self.label_dic[x] for x in train_labels]
        test_labels = [self.label_dic[x] for x in test_labels]
        # train_labels = np.array(train_labels)
        # test_labels = np.array(test_labels)
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        #赋值
        self.train_img = train_images
        self.test_img= test_images
        self.train_labels = train_labels
        self.test_labels = test_labels

class CNN(object):
    def __init__(self,out_num=None,input_shape=None):
        # 模型初始化
        model=models.Sequential()
        model.add(layers.Conv2D(filters=32,kernel_size=[3,3],padding='same',activation=tf.nn.relu,input_shape=input_shape))
        model.add(layers.Conv2D(filters=32,kernel_size=[3,3],activation=tf.nn.relu))
        model.add(layers.MaxPool2D(pool_size=[2,2]))
        model.add(layers.Conv2D(filters=64,kernel_size=[3,3],padding='same',activation=tf.nn.relu))
        model.add(layers.Conv2D(filters=64, kernel_size=[3,3],activation=tf.nn.relu))
        model.add(layers.MaxPool2D(pool_size=[2,2]))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=512,activation=tf.nn.relu))
        model.add(layers.Dense(units=out_num,activation='softmax'))
        model.summary()
        self.cnn=model

    def train(self,train_images,train_labels,test_images,test_labels):
        check_path='./ckpt/cp-{epoch:04d}.ckpt'
        # period 每隔5epoch保存一次，断点续训
        #ModelCheckpoint回调与使用model.fit（）进行的训练结合使用，
        #可以稍后加载模型或权重以从保存的状态继续训练。
        save_model_cb=tf.keras.callbacks.ModelCheckpoint(check_path,save_weights_only=True,verbose=1,period=5)
        # 编译上述构建好的神经网络模型
        # 指定优化器为 adam
        # 制定损失函数为交叉熵损失
        self.cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        # 指定训练特征集和训练标签集
        self.cnn.fit(train_images,train_labels,epochs=5,callbacks=[save_model_cb],batch_size=64)
        # 在测试集上进行模型评估
        test_loss,test_acc=self.cnn.evaluate(test_images,test_labels)
        print("准确率: %.4f，共验证了%d张图片 " % (test_acc,len(test_labels)))

if __name__ == '__main__':
    data=Data()
    face_cnn=CNN(data.len_uni_labels,data.input_shape)
    face_cnn.train(data.train_img,data.train_labels,data.test_img,data.test_labels)



