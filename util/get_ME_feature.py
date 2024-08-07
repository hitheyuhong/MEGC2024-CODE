
import os
import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import math
import pandas as pd
import torch
import argparse
detector = dlib.get_frontal_face_detector() #获取人脸分类器
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')    # 获取人脸检测器
# Dlib 检测器和预测器
font = cv2.FONT_HERSHEY_SIMPLEX
landmark0=[]
def crop_picture(img_rd,size):
    # print(img_rd.shape)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    #   人脸数
    faces = detector(img_gray, 0)
    # 标 68 个点
    for i in range(len(faces)):
        # 取特征点坐标
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
    #两个眼角的位置
    left=landmarks[39]
    right=landmarks[42]
    # print(left)
    # print(right)
    thete=math.atan(float(right[0,1]-left[0,1])/(right[0,0]-left[0,0]))
    # print("角度是{}".format(thete))

    gezi=int((right[0,0]-left[0,0])/2)
    center=[int((right[0,0]+left[0,0])/2),int((right[0,1]+left[0,1])/2)]
    # 从中心点阔9格
    cv2.rectangle(img_rd, (center[0] - int(4.5 * gezi), center[1] - int(3.5 * gezi)), (center[0] + int(4.5 * gezi), center[1] + int(5.5 * gezi)),
                  (0, 0, 255), 2)

    a=(center[1] - int(3 * gezi))
    b=center[1] +int(5 * gezi)
    c=(center[0] - int(4 * gezi))
    d=center[0] +int(4 * gezi)

    a=max((center[1] - int(3 * gezi)),0)
    # b=min(center[1] +int(5.5 * gezi),399)
    c=max(center[0] - int(4 * gezi),0)
    # d=min(center[0] +int(4.5 * gezi),399)
    img_crop = img_rd[a:b, c:d]

    img_crop_samesize = cv2.resize(img_crop, (size, size))
    return landmarks, img_crop_samesize, a, b, c, d

def get_roi_bound(low,high,round,landmark0):

    roi1_points = landmark0[low:high]
    #print(roi1_points)

    roi1_high = roi1_points[:, 0].argmax(axis=0)
    roi1_low = roi1_points[:, 0].argmin(axis=0)
    roi1_left = roi1_points[:, 1].argmin(axis=0)
    roil_right = roi1_points[:, 1].argmax(axis=0)

    roil_h = roi1_points[roi1_high, 0]
    roi1_lo = roi1_points[roi1_low, 0]
    roi1_le = roi1_points[roi1_left, 1]
    roil_r = roi1_points[roil_right, 1]

    roil_h_ex = (roil_h + round)[0, 0]
    roi1_lo_ex = (roi1_lo - round)[0, 0]
    roi1_le_ex = (roi1_le - round)[0, 0]
    roil_r_ex = (roil_r + round)[0, 0]
    return (roil_h_ex),(roi1_lo_ex),(roi1_le_ex),(roil_r_ex)
def get_roi(flow,percent):

    r1, theta1 = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    r1=np.ravel(r1)

    x1=np.ravel(flow[:, :, 0])
    y1=np.ravel(flow[:, :, 1])

    arg=np.argsort(r1)  #代表了r1这个矩阵内元素的从小到大顺序
    num=int(len(r1)*(1-percent))
    x_new=0
    y_new=0

    for i in range(num,len(arg)):#想取相对比较大的
        a=arg[i]
        x_new+=x1[a]
        y_new+=y1[a]
    x = x_new/(len(r1)*percent)
    y = y_new/(len(r1)*percent)
    return x,y
# 返回图像的68个标定点
def tu_landmarks(gray,img_rd,landmark0,frame_shang,frame_left):
    faces = detector(gray, 0)
    if(len(faces)==0):
        landmark0[:, 0]=landmark0[:,0]-frame_left
        landmark0[:, 1]=landmark0[:,1]-frame_shang
        landmarkss=landmark0
    else:
        landmarkss = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[0]).parts()])
    return landmarkss


#获得FP的flow
def getcasmelong(video_path,start,end):
    #读视频文件夹
    
    fileList1 = os.listdir(video_path)  # 图片路径
    fileList1.sort(key=lambda x: int(x[3:-4]))  # 对提取的图片排序
    
    # 根据视频分辨率，选取实际使用的
    fileList = []
    l = 0
    for i in fileList1:
        if (int(i[3:-4])<=end and int(i[3:-4])>=start ):
            fileList.append(i)
        l = l + 1
    try:

        flow=getflow(video_path,fileList)
        flow=torch.from_numpy(flow)
        return flow
    except:
        # print("计算光流出错")
        return torch.zeros((18, 20, 2))


       
def getcasme2(video_path,start,end,num):
    #读视频文件夹
    
    fileList1 = os.listdir(video_path)  # 图片路径
    fileList1.sort(key=lambda x: int(x[3:-4]))  # 对提取的图片排序
    
    start=max(start-num,0)
    
    # 根据视频分辨率，选取实际使用的
    fileList = []
    l = 0
    for i in fileList1:
        if (int(i[3:-4])<=end and int(i[3:-4])>=start and l%7==0):
            fileList.append(i)
        l = l + 1
    try:

        flow = getflow(video_path, fileList)
        flow = torch.from_numpy(flow)
        return flow
    except:
        # print("计算光流出错")
        return torch.zeros((18, 20, 2))
    
def getsammlong(video_path,start,end):
    #读视频文件夹
    
    fileList1 = os.listdir(video_path)  # 图片路径
    fileList1.sort(key=lambda x: int(x[6:-4]))  # 对提取的图片排序
    
    start=max(start-5,0)

    # 根据视频分辨率，选取实际使用的
    fileList = []
    l = 0

    for i in fileList1:
        if (int(i[6:-4])<=end and int(i[6:-4])>=start  and l%7==0):
            fileList.append(i)
        l = l + 1
    try:
        flow = getflow(video_path, fileList)
        flow = torch.from_numpy(flow)
        return flow
    except:
        # print("计算光流出错")
        return torch.zeros((18, 20, 2))

def getflow(video_path,fileList):
    if(True):
        k = 0
        for i in fileList:
            k = k + 1
            if (k == 1):
                flow1_total = [[0, 0]]  # 是存储了不同位置帧之间的光流
                flow1_total1 = [[0, 0]]
                flow1_total2 = [[0, 0]]
                flow1_total3 = [[0, 0]]
                flow2_total = [[0, 0]]
                flow3_total = [[0, 0]]
                flow3_total1 = [[0, 0]]
                flow3_total2 = [[0, 0]]
                flow3_total3 = [[0, 0]]
                flow4_total = [[0, 0]]
                flow4_total1 = [[0, 0]]
                flow4_total2 = [[0, 0]]
                flow4_total3 = [[0, 0]]
                flow4_total4 = [[0, 0]]
                flow4_total5 = [[0, 0]]
                flow5_total1 = [[0, 0]]
                flow5_total2 = [[0, 0]]
                flow2_total1 = [[0, 0]]
                flow6_total = [[0, 0]]
                flow7_total = [[0, 0]]

                img_rd = cv2.imread(video_path + i)  # D:/face_image_test/EP07_04/

                landmark0, img_rd, frame_shang, frame_xia, frame_left, frame_right = crop_picture(img_rd, 256)
                # 记录框的位置，上下左右在整个图片中的坐标，和68点的位置。img_rd是被裁减之后的面部位置，并resize到256*256

                gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)  # 变成灰度图
                landmark0 = tu_landmarks(gray, img_rd, landmark0, frame_shang, frame_left)  # 对人脸68个点的定位
                # 相对与新图片的68点的位置。

                round1 = 0
                roil_right, roi1_left, roi1_low, roi1_high = get_roi_bound(17, 22, 0, landmark0)  # 左眉毛的位置
                # cv2.rectangle(img_rd, (roi1_left-5, roi1_low - 15), (roil_right, roi1_high + 5), (0, 255, 0), 1)

                roi1_sma = []  # 存储了左眼的三个小的感兴趣区域，从里到外
                roi1_sma.append([landmark0[20, 1] - (roi1_low - 15), landmark0[20, 0] - (roi1_left - 5)])
                roi1_sma.append([landmark0[19, 1] - (roi1_low - 15), landmark0[19, 0] - (roi1_left - 5)])
                roi1_sma.append([landmark0[18, 1] - (roi1_low - 15), landmark0[18, 0] - (roi1_left - 5)])
                cv2.rectangle(img_rd, (landmark0[20, 0] - 10, landmark0[20, 1] + 10),
                              (landmark0[20, 0] + 10, landmark0[20, 1] - 10), (0, 255, 255), 1)
                cv2.rectangle(img_rd, (landmark0[19, 0] - 10, landmark0[19, 1] + 10),
                              (landmark0[19, 0] + 10, landmark0[19, 1] - 10), (0, 255, 255), 1)
                cv2.rectangle(img_rd, (landmark0[18, 0] - 10, landmark0[18, 1] + 10),
                              (landmark0[18, 0] + 10, landmark0[18, 1] - 10), (0, 255, 255), 1)
                prevgray_roi1 = gray[(roi1_low - 15):roi1_high + 5, roi1_left - 5:roil_right]

                # 右眼
                roi3_right, roi3_left, roi3_low, roi3_high = get_roi_bound(22, 27, 0, landmark0)
                # cv2.rectangle(img_rd, (roi3_left, roi3_high + 5), (roi3_right, roi3_low - 15), (0, 255, 0), 1)
                roi3_sma = []  # 存储了右眼的三个小的感兴趣区域，从里到外
                roi3_sma.append([landmark0[23, 1] - (roi3_low - 15), landmark0[23, 0] - roi3_left])
                roi3_sma.append([landmark0[24, 1] - (roi3_low - 15), landmark0[24, 0] - roi3_left])
                roi3_sma.append([landmark0[25, 1] - (roi3_low - 15), landmark0[25, 0] - roi3_left])
                cv2.rectangle(img_rd, (landmark0[25, 0] - 10, landmark0[25, 1] + 10),
                              (landmark0[25, 0] + 10, landmark0[25, 1] - 10), (0, 255, 255), 1)
                cv2.rectangle(img_rd, (landmark0[24, 0] - 10, landmark0[24, 1] + 10),
                              (landmark0[24, 0] + 10, landmark0[24, 1] - 10), (0, 255, 255), 1)
                cv2.rectangle(img_rd, (landmark0[23, 0] - 10, landmark0[23, 1] + 10),
                              (landmark0[23, 0] + 10, landmark0[23, 1] - 10), (0, 255, 255), 1)
                prevgray_roi3 = gray[(roi3_low - 15):roi3_high + 5, roi3_left:roi3_right]
                # print(prevgray_roi1.shape)

                # 嘴巴处的四个
                roi4_right, roi4_left, roi4_low, roi4_high = get_roi_bound(48, 67, 0, landmark0)
                # cv2.rectangle(img_rd, (roi4_left-20, roi4_high + 10), (roi4_right+20, roi4_low - 15), (0, 255, 0), 1)
                roi4_sma = []
                roi4_sma.append([landmark0[48, 1] - (roi4_low - 15), landmark0[48, 0] - (roi4_left - 20)])
                roi4_sma.append([landmark0[54, 1] - (roi4_low - 15), landmark0[54, 0] - (roi4_left - 20)])
                roi4_sma.append([landmark0[51, 1] - (roi4_low - 15), landmark0[51, 0] - (roi4_left - 20)])
                roi4_sma.append([landmark0[57, 1] - (roi4_low - 15), landmark0[57, 0] - (roi4_left - 20)])
                roi4_sma.append([landmark0[62, 1] - (roi4_low - 15), landmark0[62, 0] - (roi4_left - 20)])
                cv2.rectangle(img_rd, (landmark0[48, 0] - 10, landmark0[48, 1] + 10),
                              (landmark0[48, 0] + 10, landmark0[48, 1] - 10), (0, 255, 255), 1)
                cv2.rectangle(img_rd, (landmark0[51, 0] - 10, landmark0[51, 1] + 10),
                              (landmark0[51, 0] + 10, landmark0[51, 1] - 10), (0, 255, 255), 1)
                cv2.rectangle(img_rd, (landmark0[54, 0] - 10, landmark0[54, 1] + 10),
                              (landmark0[54, 0] + 10, landmark0[54, 1] - 10), (0, 255, 255), 1)
                cv2.rectangle(img_rd, (landmark0[57, 0] - 10, landmark0[57, 1] + 10),
                              (landmark0[57, 0] + 10, landmark0[57, 1] - 10), (0, 255, 255), 1)
                # cv2.rectangle(img_rd, (landmark0[62, 0] - 10, landmark0[62, 1] + 10),
                #               (landmark0[62, 0] + 10, landmark0[62, 1] - 10), (0, 255, 0), 1)
                prevgray_roi4 = gray[(roi4_low - 15):roi4_high + 10, roi4_left - 20:roi4_right + 20]

                # 鼻子两侧
                roi5_right, roi5_left, roi5_low, roi5_high = get_roi_bound(30, 36, 0, landmark0)
                # cv2.rectangle(img_rd, (roi5_left-22, roi5_high + 5), (roi5_right+22, roi5_low - 20), (0, 255, 0), 1)
                roi5_sma = []
                roi5_sma.append([landmark0[31, 1] - (roi5_low - 20), landmark0[31, 0] - (roi5_left - 30)])
                roi5_sma.append([landmark0[35, 1] - (roi5_low - 20), landmark0[35, 0] - (roi5_left - 30)])

                cv2.rectangle(img_rd, (landmark0[31, 0] - 20, landmark0[31, 1] + 5),
                              (landmark0[31, 0] + 10, landmark0[31, 1] - 20), (0, 255, 255), 1)
                cv2.rectangle(img_rd, (landmark0[35, 0] - 10, landmark0[35, 1] + 5),
                              (landmark0[35, 0] + 20, landmark0[35, 1] - 20), (0, 255, 255), 1)

                prevgray_roi5 = gray[(roi5_low - 20):roi5_high + 5, roi5_left - 30:roi5_right + 30]

                # 左眼睑部位
                roi6_right, roi6_left, roi6_low, roi6_high = get_roi_bound(36, 42, 0, landmark0)
                width = roi6_right - roi6_left
                height = width / 2
                xin = (roi6_high + roi6_low) / 2
                roi6_high = int(xin + 3 * height / 2)
                roi6_low = int(xin + height / 2)
                prevgray_roi6 = gray[roi6_low:roi6_high, roi6_left:roi6_right]
                cv2.rectangle(img_rd, (roi6_left, roi6_high), (roi6_right, roi6_low), (0, 255, 255), 1)

                # 右眼睑部位
                roi7_right, roi7_left, roi7_low, roi7_high = get_roi_bound(42, 48, 0, landmark0)
                # print(roi6_high)//100
                # print(roi6_low)//90
                width = roi7_right - roi7_left
                height = width / 2
                xin = (roi7_high + roi7_low) / 2
                roi7_high = int(xin + 3 * height / 2)
                roi7_low = int(xin + height / 2)
                prevgray_roi7 = gray[roi7_low:roi7_high, roi7_left:roi7_right]
                cv2.rectangle(img_rd, (roi7_left, roi7_high), (roi7_right, roi7_low), (0, 255, 255), 1)

                roi2_right, roi2_left, roi2_low, roi2_high = get_roi_bound(29, 31, 13, landmark0)
                prevgray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]

                # cv2.rectangle(img_rd, (roi2_left + round1, roi2_high + 5 - round1),
                #               (roi2_right - round1, roi2_low - 10 + round1), (0, 255, 0), 1)
                # cv2.rectangle(img_rd, (roi2_left + 5, roi2_high + 5 - 15), (roi2_right - 5, roi2_low - 10 + 25),
                #               (0, 255, 0), 1)
                # path33 = "D:/dataset/micro_datatset/imageforpaper/img2.jpg"
                # cv2.imwrite(path33, img_rd)
                # cv2.imshow("image1", img_rd)
                # cv2.waitKey(0)
                # print("ll")
                # print(prevgray_roi2.shape)


            else:

                if (True):

                    img_rd1 = cv2.imread(video_path + i)  # D:/face_image_test/EP07_04/
                    img_crop = img_rd1[frame_shang:frame_xia, frame_left:frame_right]  # 按照第一个图的框切割出一个脸

                    img_rd = cv2.resize(img_crop, (256, 256))
                    gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
                    # 求全局的光流
                    gray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]
                    # 使用Gunnar Farneback算法计算密集光流
                    flow2 = cv2.calcOpticalFlowFarneback(prevgray_roi2, gray_roi2, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                    flow2 = np.array(flow2)

                    # him2, x1, y1 = get_roi_him(flow2[15:-10, 5:-5, :])
                    x1, y1 = get_roi(flow2[15:-10, 5:-5, :], 0.7)
                    # print("全局运动为{}and{}".format(x1,y1))
                    flow2_total1.append([x1, y1])

                    # 进行面部对齐，移动切割框
                    l = 0
                    while ((x1 ** 2 + y1 ** 2) > 1):  # 移动比较大，相应移动脸的位置
                        l = l + 1
                        if (l > 3):
                            print("ppp")
                            break
                        frame_left += int(round(x1))
                        frame_shang += int(round(y1))
                        frame_right += int(round(x1))
                        frame_xia += int(round(y1))
                        # print(frame_left)
                        # print(frame_shang)
                        # print(frame_right)
                        # print(frame_xia)
                        frame_left = max(0, frame_left)
                        frame_shang = max(0, frame_shang)
                        img_rd1 = cv2.imread(video_path + i)

                        img_crop = img_rd1[frame_shang:frame_xia, frame_left:frame_right]
                        img_rd = cv2.resize(img_crop, (256, 256))
                        gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
                        # 求全局的光流
                        gray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]
                        # 使用Gunnar Farneback算法计算密集光流
                        flow2 = cv2.calcOpticalFlowFarneback(prevgray_roi2, gray_roi2, None, 0.5, 3, 15, 5, 7, 1.5,
                                                             0)
                        flow2 = np.array(flow2)

                        # him2, x1, y1 = get_roi_him(flow2[15:-10, 5:-5, :])
                        x1, y1 = get_roi(flow2[15:-10, 5:-5, :], 0.7)

                        # print("全局运动为{}and{}".format(x1, y1))
                        flow2_total1.append([x1, y1])
                    # 对齐完毕

                    gray_roi1 = gray[(roi1_low - 15):roi1_high + 5, roi1_left - 5:roil_right]
                    # 使用Gunnar Farneback算法计算密集光流
                    try:
                        flow1 = cv2.calcOpticalFlowFarneback(prevgray_roi1, gray_roi1, None, 0.5, 3, 15, 5, 7, 1.5,
                                                             0)  # 计算整个左眉毛处的光流
                    except:
                        break
                    flow1[:, :, 0] = flow1[:, :, 0]
                    flow1[:, :, 1] = flow1[:, :, 1]
                    # print("pppppp")
                    round1 = 10
                    roi1_sma = np.array(roi1_sma)
                    # print(roi1_sma)
                    a, b = get_roi(flow1[round1:-round1, round1:-round1, :], 0.2)  # 去掉光流特征矩阵周边round大小的部分，求均值
                    a1, b1 = get_roi(  # 一个感兴趣区域处的平均光流
                        flow1[roi1_sma[0, 0] - 10:roi1_sma[0, 0] + 10, roi1_sma[0, 1] - 10:roi1_sma[0, 1] + 10, :],
                        0.2)
                    a2, b2 = get_roi(
                        flow1[roi1_sma[1, 0] - 10:roi1_sma[1, 0] + 10, roi1_sma[1, 1] - 10:roi1_sma[1, 1] + 10, :],
                        0.2)
                    a3, b3 = get_roi(
                        flow1[roi1_sma[2, 0] - 10:roi1_sma[2, 0] + 10, roi1_sma[2, 1] - 10:roi1_sma[2, 1] + 10, :],
                        0.2)

                    flow1_total1.append([a1 - x1, b1 - y1])  # 局部区域减去全局光流
                    flow1_total2.append([a2 - x1, b2 - y1])
                    flow1_total3.append([a3 - x1, b3 - y1])
                    flow1_total.append([a - x1, b - y1])

                    gray_roi3 = gray[(roi3_low - 15):roi3_high + 5, roi3_left:roi3_right]
                    # 使用Gunnar Farneback算法计算密集光流
                    flow3 = cv2.calcOpticalFlowFarneback(prevgray_roi3, gray_roi3, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                    flow3[:, :, 0] = flow3[:, :, 0]
                    flow3[:, :, 1] = flow3[:, :, 1]
                    round1 = 10
                    # a = np.mean(flow3[round1:-round1, round1:-round1, 0])
                    # b = np.mean(flow3[round1:-round1, round1:-round1, 1])
                    roi3_sma = np.array(roi3_sma)
                    # print(roi1_sma)
                    a, b = get_roi(flow3[round1:-round1, round1:-round1, :], 0.3)
                    a1, b1 = get_roi(
                        flow3[roi3_sma[0, 0] - 10:roi3_sma[0, 0] + 10, roi3_sma[0, 1] - 10:roi3_sma[0, 1] + 10, :],
                        0.3)
                    a2, b2 = get_roi(
                        flow3[roi3_sma[1, 0] - 10:roi3_sma[1, 0] + 10, roi3_sma[1, 1] - 10:roi3_sma[1, 1] + 10, :],
                        0.3)
                    a3, b3 = get_roi(
                        flow3[roi3_sma[2, 0] - 10:roi3_sma[2, 0] + 10, roi3_sma[2, 1] - 10:roi3_sma[2, 1] + 10, :],
                        0.3)

                    flow3_total1.append([a1 - x1, b1 - y1])
                    flow3_total2.append([a2 - x1, b2 - y1])
                    flow3_total3.append([a3 - x1, b3 - y1])
                    flow3_total.append([a - x1, b - y1])

                    gray_roi4 = gray[(roi4_low - 15):roi4_high + 10, roi4_left - 20:roi4_right + 20]
                    # print(gray_roi4.shape)
                    # print(prevgray_roi4.shape)
                    # 使用Gunnar Farneback算法计算密集光流
                    flow4 = cv2.calcOpticalFlowFarneback(prevgray_roi4, gray_roi4, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                    flow4[:, :, 0] = flow4[:, :, 0]
                    flow4[:, :, 1] = flow4[:, :, 1]
                    round1 = 10
                    roi4_sma = np.array(roi4_sma)
                    # print(roi1_sma)
                    a, b = get_roi(flow4[round1:-round1, round1:-round1, :], 0.3)
                    a1, b1 = get_roi(
                        flow4[roi4_sma[0, 0] - 10:roi4_sma[0, 0] + 10, roi4_sma[0, 1] - 10:roi4_sma[0, 1] + 20, :],
                        0.2)
                    a2, b2 = get_roi(
                        flow4[roi4_sma[1, 0] - 10:roi4_sma[1, 0] + 10, roi4_sma[1, 1] - 20:roi4_sma[1, 1] + 10, :],
                        0.2)
                    a3, b3 = get_roi(
                        flow4[roi4_sma[2, 0] - 10:roi4_sma[2, 0] + 10, roi4_sma[2, 1] - 10:roi4_sma[2, 1] + 10, :],
                        0.2)
                    a4, b4 = get_roi(
                        flow4[roi4_sma[3, 0] - 10:roi4_sma[3, 0] + 10, roi4_sma[3, 1] - 10:roi4_sma[3, 1] + 10, :],
                        0.2)
                    a5, b5 = get_roi(
                        flow4[roi4_sma[4, 0] - 10:roi4_sma[4, 0] + 10, roi4_sma[4, 1] - 10:roi4_sma[4, 1] + 10, :],
                        0.2)

                    flow4_total1.append([a1 - x1, b1 - y1])
                    flow4_total2.append([a2 - x1, b2 - y1])
                    flow4_total3.append([a3 - x1, b3 - y1])
                    flow4_total4.append([a4 - x1, b4 - y1])
                    flow4_total5.append([a5 - x1, b5 - y1])
                    flow4_total.append([a - x1, b - y1])

                    gray_roi5 = gray[(roi5_low - 20):roi5_high + 5, roi5_left - 30:roi5_right + 30]
                    # 使用Gunnar Farneback算法计算密集光流
                    flow5 = cv2.calcOpticalFlowFarneback(prevgray_roi5, gray_roi5, None, 0.5, 3, 15, 5, 7, 1.5, 0)

                    round1 = 10
                    roi5_sma = np.array(roi5_sma)
                    # print(roi1_sma)
                    # print("=========")
                    # print(roi5_sma)
                    # print(flow5.shape)
                    # print(roi5_sma)
                    #
                    # print(flow5[roi5_sma[0, 0] - 25:roi5_sma[0, 0] + 5, roi5_sma[0, 1] - 20:roi5_sma[0, 1] + 10,
                    #       :].shape)
                    # print(flow5[roi5_sma[1, 0] - 25:roi5_sma[1, 0] + 5, roi5_sma[1, 1] - 20:roi5_sma[1, 1] + 10,
                    #       :].shape)
                    a1, b1 = get_roi(
                        flow5[roi5_sma[0, 0] - 20:roi5_sma[0, 0] + 5, roi5_sma[0, 1] - 20:roi5_sma[0, 1] + 10, :],
                        0.2)

                    a2, b2 = get_roi(
                        flow5[roi5_sma[1, 0] - 20:roi5_sma[1, 0] + 5, roi5_sma[1, 1] - 10:roi5_sma[1, 1] + 20, :],
                        0.2)

                    flow5_total1.append([a1 - x1, b1 - y1])
                    flow5_total2.append([a2 - x1, b2 - y1])
                    round1 = 5

                    gray_roi6 = gray[roi6_low:roi6_high, roi6_left:roi6_right]
                    gray_roi7 = gray[roi7_low:roi7_high, roi7_left:roi7_right]
                    flow6 = cv2.calcOpticalFlowFarneback(prevgray_roi6, gray_roi6, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                    flow7 = cv2.calcOpticalFlowFarneback(prevgray_roi7, gray_roi7, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                    a1, b1 = get_roi(flow6[round1:-round1, round1:-round1, :], 0.3)
                    a2, b2 = get_roi(flow7[round1:-round1, round1:-round1, :], 0.3)
                    flow6_total.append([a1 - x1, b1 - y1])
                    flow7_total.append([a2 - x1, b2 - y1])


            if (k == len(fileList)):

                flow = np.stack((np.array(flow1_total),np.array(flow1_total1),np.array(flow1_total2),np.array(flow1_total3),np.array(flow3_total),np.array(flow3_total1),np.array(flow3_total2)
                                 ,np.array(flow3_total3),np.array(flow4_total),np.array(flow4_total1),np.array(flow4_total2),np.array(flow4_total3),np.array(flow4_total4)
                                 ,np.array(flow4_total5),np.array(flow5_total1),np.array(flow5_total2),np.array(flow6_total),np.array(flow7_total)))
                return flow














