# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2023/7/12 10:55
@version: 1.0
@File: face_find.py
'''

from scipy.spatial import distance as dist
import numpy as np
import dlib
import cv2
from imutils import face_utils
import pyttsx3



def eye_aspect_ratio(eye):
    # 计算距离，竖直的
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算距离，水平的
    C = dist.euclidean(eye[0], eye[3])
    # ear值
    ear = (A + B) / (2.0 * C)
    return ear


def main():
    pt = pyttsx3.init()
    FACIAL_LANDMARKS_68_IDXS = dict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17))
    ])
    # 设置判断参数
    EYE_AR_THRESH = 0.2  # ear小于0.3判断为闭眼
    EYE_AR_CONSEC_FRAMES = 10  # 连续三帧ear都小于0.3判断为眨眼
    # 初始化计数器
    COUNTER = 0
    TOTAL = 0

    detector = dlib.get_frontal_face_detector()  # 人脸检测
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 关键点检测

    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

    capfrom = cv2.VideoCapture(0)
    num=0
    # 遍历每一帧
    while True:
        # 预处理
        _,frame = capfrom.read()
        if frame is None:
            break
            # 按比例缩放图像尺寸，这个步骤对检测效果有影响，越大越慢。
        (h, w) = frame.shape[:2]
        width = 1200
        r = width / float(w)
        dim = (width, int(h * r))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸 返回的检测到的人脸位置
        rects = detector(gray, 0)
        # 接着我们遍历每一个检测到的人脸 ，分别对每一张脸做关键点检测，ears值计算。
        for rect in rects:
            # 获取坐标
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # 分别计算ear值
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # 算一个平均的
            ear = (leftEAR + rightEAR) / 2.0
            # 绘制眼睛区域
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # 检查是否满足阈值
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                # 如果连续几帧都是闭眼的，总数算一次
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    print('检测到闭眼时间超过阈值')
                    pt.say('检测到闭眼时间超过阈值')
                    pt.runAndWait()
                    # 重置
                COUNTER = 0
                # 显示 把眨眼的次数显示在屏幕上
            cv2.putText(frame, "count={},tired times: {}".format(COUNTER,TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # 展示图像
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
    vs.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
