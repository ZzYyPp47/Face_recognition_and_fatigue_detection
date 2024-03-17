# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2023/7/12 9:14
@version: 1.0
@File: face.py
'''


import tensorflow as tf
import cv2
import numpy as np
from facecnn import CNN,Data
import datetime
from PIL import Image, ImageDraw, ImageFont


class Predict(object):
    def __init__(self,out_num=None,input_shape=None):
        latest = tf.train.latest_checkpoint('./ckpt')
        self.data = Data()
        if out_num is None:
            out_num=self.data.len_uni_labels
        if input_shape is None:
            input_shape=self.data.input_shape
        self.model = CNN(out_num,input_shape)
        # 恢复网络权重
        self.model.cnn.load_weights(latest)
        self.face_find()

    def predict(self,img=None,shape=None):
        # 初始化
        if img is None:
            print('found no img to predict,execution exit!')
            return None
        if shape is None:
            shape=self.data.input_shape
        # 以黑白方式读取图片
        img = cv2.resize(img,(64,64)) / 255.
        x = np.reshape(img, (1, self.data.input_shape[0], self.data.input_shape[1], self.data.input_shape[2]))
        y = self.model.cnn.predict(x,verbose=0)
        return y


    def face_find(self,camera_idx=0,classfier_path='D:\\python\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml'):
        # 视频捕获(选择idx)
        capfrom = cv2.VideoCapture(camera_idx)
        # 选择分类器
        classfier = cv2.CascadeClassifier(classfier_path)
        # 矩形框颜色(红色)
        color = (0, 0, 255)
        #循环捕获
        while (capfrom.isOpened()):
            ret, img = capfrom.read()
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = classfier.detectMultiScale(img_gray, scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
            if len(faces)>0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h),color, 2)
                    pred=self.predict(img_gray[y:y + h, x:x + w])
                    text='{}已打卡于时间：{:%Y-%m-%d %H:%M:%S}'.format(self.data.re_label_dic[np.argmax(pred)],datetime.datetime.now())
                    print(text)
                    for ii in range(pred.shape[1]):
                        text='{}:{:.2f}%'.format(self.data.re_label_dic[ii],pred[0][ii]*100)
                        img=self.Draw_I(img,text,x,y-20*(ii+1))
            cv2.imshow('face_find', img)
            c = cv2.waitKey(1)
            if c == 27:  # Esc 键
                break
        capfrom.release()
        cv2.destroyAllWindows()


    def Draw_I(self,img, text, left, top, textColor=(255, 0, 0), textSize=20):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 创建一个可以在给定图像上绘图的对象
            draw = ImageDraw.Draw(img)
            # 字体的格式
            fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
            # 绘制文本
            draw.text((left, top), text, textColor, font=fontStyle)
            # 转换回OpenCV格式
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    face=Predict()