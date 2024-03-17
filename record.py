# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2023/7/12 9:32
@version: 1.0
@File: record.py
'''


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def CatchFromVideo(name,save_path,num_max=200,camera_idx=0,classfier_path='D:\\python\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml'):
    #调用摄像头
    capfrom = cv2.VideoCapture(camera_idx)
    #调用分类器
    classfier = cv2.CascadeClassifier(classfier_path)
    #选择颜色
    color = (0, 0, 255)
    #初始化num数
    num=1
    while (capfrom.isOpened()):
        ret, img = capfrom.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = classfier.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_name='{}{}_{}.png'.format(save_path,str(num),name)
                face=img[y-10:y+h+10,x-10:x+w+10]
                cv2.imwrite(face_name,face)
                print('save successed:To ->'+face_name)
                num+=1
                if num > num_max:
                    break
                text='已保存{}张来自{}的面部数据'.format(str(num),name)
                img=Draw_I(img,text,0,0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            if num > num_max:
                break
            cv2.imshow('face_find', img)
            cv2.waitKey(1)
    capfrom.release()
    cv2.destroyAllWindows()


def Draw_I(img, text, left, top, textColor=(255,0,0),textSize=25):
     if (isinstance(img, np.ndarray)): # 判断是否OpenCV图片类型
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
    name=input('请输入姓名(英文):')
    save_path='./face_list/'+name+'/'
    os.mkdir(save_path)
    CatchFromVideo(name=name,save_path=save_path)

