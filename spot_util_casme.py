import util.sim_filter as sim_filter
import os
import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import math

import matplotlib.pyplot as plt
import util.try_emd as try_emd
import xlrd
import xlwt
from xlrd import xldate_as_tuple
from scipy.signal import find_peaks,peak_widths
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
    # print(8*gezi)

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


#对给定的每个视频帧之间的光流。进行求平方和和开根号的计算，并画出动作线
def draw_line(flow_total):
    flow_total=np.array(flow_total)

    flow_total=np.sum(flow_total**2,axis=1)
    flow_total=np.sqrt(flow_total)

    return flow_total
def fenxi(flow_total, imf_sum1,yuzhi1,yuzhi2):
    flow_total = np.array(flow_total)
    low=np.min(flow_total)
    flow_total=flow_total-low
    flow_total_fenxi=[]
    for j in range(len(flow_total)):
        if(flow_total[j]>=yuzhi1):
            flow_total_fenxi.append(j)
    flow_total_pp = []
    if(len(flow_total_fenxi)>0):
        start=flow_total_fenxi[0]
        end=flow_total_fenxi[0]
        st=0
        for i in range(len(flow_total_fenxi)):
            if(flow_total_fenxi[i]>=end and flow_total_fenxi[i]-end<5):
                end=flow_total_fenxi[i]
            else:
                flow_total_pp.append([start,end])
                start = flow_total_fenxi[i]
                end = flow_total_fenxi[i]
        flow_total_pp.append([start, end])
    flow_total_fenxi = []
    flow_total_pp=np.array(flow_total_pp)
    for i in range(len(flow_total_pp)):
        start=flow_total_pp[i,0]
        end = flow_total_pp[i, 1]
        # start_new=min(0,start-25)
        # end_new=max(len(flow_total-1),end+25)
        # low = np.min(flow_total[start_new:end_new])
        # flow_total[start:end]=flow_total[start:end]-low
        for j in range(start,end):
            a = max(0, j - 30)
            b=min(len(flow_total-1),j+30)
            low=np.min(flow_total[a:b])

            if(flow_total[j]-low>yuzhi2 ):
                flow_total_fenxi.append(j)
    flow_total_pp2 = []
    if (len(flow_total_fenxi) > 0):
        start = flow_total_fenxi[0]
        end = flow_total_fenxi[0]
        st = 0
        for i in range(len(flow_total_fenxi)):
            if (flow_total_fenxi[i] >= end and flow_total_fenxi[i] - end < 5):
                end = flow_total_fenxi[i]
            else:
                flow_total_pp2.append([start, end])
                start = flow_total_fenxi[i]
                end = flow_total_fenxi[i]
        flow_total_pp2.append([start, end])

    return np.array(flow_total_pp2)
def fenxi1(flow_total,imf_sum1,yuzhi1,yuzhi2): #使用寻找峰的方法
    flow_total = np.array(flow_total)
    low=np.min(flow_total)    #找到最小值
    flow_total=flow_total-low     #从零开始
    flow_total_fenxi=[]
    for j in range(len(flow_total)):#找到大于较小阈值
        if(flow_total[j]>=yuzhi1):
            flow_total_fenxi.append(j)   #大于较小阈值的帧的索引
    flow_total_pp = []
    if(len(flow_total_fenxi)>0): #对经过第一步筛选的，帧相邻的连在一起
        start=flow_total_fenxi[0]
        end=flow_total_fenxi[0]
        st=0
        for i in range(len(flow_total_fenxi)):
            if(flow_total_fenxi[i]>=end and flow_total_fenxi[i]-end<3):
                end=flow_total_fenxi[i]
            else:
                flow_total_pp.append([start,end])
                start = flow_total_fenxi[i]
                end = flow_total_fenxi[i]
        flow_total_pp.append([start, end])
    flow_total_fenxi = []
    flow_total_pp=np.array(flow_total_pp)

    for i in range(len(flow_total_pp)): #第二次筛选
        start=flow_total_pp[i,0]
        end = flow_total_pp[i, 1]
        # start_new=min(0,start-25)
        # end_new=max(len(flow_total-1),end+25)
        # low = np.min(flow_total[start_new:end_new])
        # flow_total[start:end]=flow_total[start:end]-low
        for j in range(start,end):
            a = max(0, j - 30)
            b  = min(len(flow_total)-1,j+30)  #找到这个点的两边，左边右边各30，注意不能超过滑动窗口的碧娜姐
            low= np.min(flow_total[a:b])     #左右区间都找最小的
            low1 = np.min(imf_sum1[a:b])     #左右区间都找最小的
            if (flow_total[j] - low > yuzhi2 and imf_sum1[j] - low1 >0.8):
                flow_total_fenxi.append(j)
    flow_total_pp2 = []
    if (len(flow_total_fenxi) > 0):
        start = flow_total_fenxi[0]
        end = flow_total_fenxi[0]
        st = 0
        for i in range(len(flow_total_fenxi)):
            if (flow_total_fenxi[i] >= end and flow_total_fenxi[i] - end < 3):
                end = flow_total_fenxi[i]
            else:
                flow_total_pp2.append([start, end])
                start = flow_total_fenxi[i]
                end = flow_total_fenxi[i]
        # start=max(start-5,0)
        # end=min(end+5,199)
        flow_total_pp2.append([start, end])

    return np.array(flow_total_pp2)

def expend(flow1_total_fenxi,flow1_total_edm):
    for i in range(len(flow1_total_fenxi)):
        start=flow1_total_fenxi[i, 0]
        end=flow1_total_fenxi[i,1]
        a1 = max(0, start - 30)
        b1 = min(len(flow1_total_edm )-1, start + 30)
        a2 = max(0,  end- 30)
        b2 = min(len(flow1_total_edm) - 1, end + 30)
        if(end>start):  #因为有可能end=start
            high=np.max(flow1_total_edm[start:end])
        else:
            high=flow1_total_edm[start]

        st_low = np.min(flow1_total_edm[a1:b1])
        st_arglow=np.argmin(flow1_total_edm[a1:b1])+a1  #start的左右中最小的索引
        en_low = np.min(flow1_total_edm[a2:b2])      #end的左右中最小的索引
        en_arglow=np.argmin(flow1_total_edm[a2:b2])+a2
        if(st_arglow<start):
            for j in range(start-1,-1,-1):
                if(flow1_total_edm[j]-st_low<0.33*(high-st_low)):
                    start=j
                    break
                if (flow1_total_edm[j] > flow1_total_edm[j + 1]):
                    start = j + 2
                    break
        else:
            left=max(start-10,0)
            aa=np.argmin(flow1_total_edm[left:start+1]) +left #代表了start左侧十个中值最小的索引
            if(flow1_total_edm[start]-flow1_total_edm[aa]>0.3):
                start=aa+1
        if (en_arglow > end):
            for j in range(end+1, en_arglow):
                if (flow1_total_edm[j] - en_low < 0.33*(high-en_low)):
                    end = j
                    break
                if(flow1_total_edm[j]>flow1_total_edm[j-1]):
                    end=j-2
                    break
        else:
            right=min(end+10,len(flow1_total_edm )-1)
            aa=np.argmin(flow1_total_edm[end:right+1])+end#代表了end右侧十个中值最小的索引
            if(flow1_total_edm[end]-flow1_total_edm[aa]>0.3):
                end=aa-1  #用最小值的索引进行替换

        flow1_total_fenxi[i, 0]=start
        flow1_total_fenxi[i, 1]=end
    return flow1_total_fenxi

def divide(flow1_total_fenxi, flow1_total_edm):
    h=[]
    for i in range(len(flow1_total_fenxi)):
        a=int(flow1_total_fenxi[i,0])
        b=int(flow1_total_fenxi[i,1])
        if((flow1_total_fenxi[i,1]-flow1_total_fenxi[i,0])>=20):
            minnum=np.argmin(flow1_total_edm[a+8:b-8])+a+8
            max1=np.max(flow1_total_edm[a:minnum])
            max2=np.max(flow1_total_edm[minnum:b])
            if((max1-flow1_total_edm[minnum]>max(0.7,0.33*max1)) and (max2-flow1_total_edm[minnum]>max(0.7,0.33*max2))):
                h.append([a,minnum-1])
                h.append([minnum+1,b])
            else:
                h.append([a,b])
        else:
            h.append([a,b])
    return np.array(h)
def draw(y1,path="",xuhao=0):
    x = np.arange(len(y1))
    plt.figure()
    plt.plot(x,y1)
    plt.title(path+str(xuhao))

    plt.show()
    # plt.savefig("D:/A/PRletterline/"+"p1/"+xuhao+AU+".jpg")
def proce2(flow1_total,yuzhi1,yuzhi2,position,xuhao,k,a,totalflow,totalflow_mic,totalflow_mac):
    fs=1
    flow1_total = draw_line(flow1_total)#作用是将光流特征转换为幅值的形式
    flow1_total = np.array(flow1_total)
    position=position+str(xuhao)+"----"  #
    flow1_total_edm1 = sim_filter.filt(flow1_total[a:-a], 1, 5, 30)  # 滤波

    hh = len(flow1_total_edm1)+2 #作用等同于200

    flow1_total_edm2,imf_sum1 = try_emd.the_emd1(flow1_total[a:-a],flow1_total_edm1, position, str(k - hh) ,fs)
    # draw(flow1_total_edm2, position + "----", k)
    flow1_total_fenxi = fenxi1(flow1_total_edm1,imf_sum1,yuzhi1,yuzhi2)  #得到了分析结果
    flow1_total_fenxi=expend(flow1_total_fenxi,flow1_total_edm1)   #向两边扩展
    flow1_total_fenxi = divide(flow1_total_fenxi, flow1_total_edm1)#将中间低的峰分成两个部分。
    flow1_total_fenxi=flow1_total_fenxi+ (k - hh) + a
    for i in range(len(flow1_total_fenxi)):
        totalflow.append(flow1_total_fenxi[i])


    return totalflow,totalflow_mic,totalflow_mac
def nms2(totalflow):
    totalflow=np.array(totalflow)
    hh=[[0,0]]
    for i in range(len(totalflow)):
        new=1
        if(i==0):
            hh=np.vstack((hh,[[totalflow[i,0],totalflow[i,1]]]))
            continue
        for j in range(1,len(hh)):
            # print("pp")
            # print(len(hh))
            #计算iou
            if(totalflow[i,0]>hh[j,1] or totalflow[i,1]<hh[j,0]):   #两个间隔完全不相交
                iou=0
            else:
                ma=max(totalflow[i,0],hh[j,0])
                mi=min(totalflow[i,1],hh[j,1])
                wid=mi-ma
                iou=max(wid/(hh[j,1]-hh[j,0]),wid/(totalflow[i,1]-totalflow[i,0]))
            #通过iou决定是不是要添加
            if(iou>0.1):#SAMM0.34  CASME 0.29   #如果重复率比较高就
                new=0
                hh[j, 1]=max(hh[j, 1],totalflow[i, 1])
                hh[j, 0]=min(hh[j, 0],totalflow[i, 0])
        if(new ==1):
            hh=np.vstack((hh, [[totalflow[i, 0], totalflow[i, 1]]]))
    return hh
def draw_roiline18(path1,path2, qian, hou,fs,casme_feature_path):  #和16的区别是,通过已经提取到的光流，调整参数，得到最优解
    # path = "D:/dataset/micro_datatset/test_casme1_face8/" + path2+"/"
    path = casme_feature_path + path2+"/"
    fileList1 = os.listdir(path)
    fileList1.sort(key=lambda x: int(x[0:hou]))
    fileList = []

    lable_vio = np.array([0, 0])
    for flowposi in fileList1:
        k=int(flowposi[0:-4] ) #这一段结束的位置
        path1=path+flowposi   #D:/dataset/micro_datatset/test_no/s24/24_0101disgustingteeth/200.npy
        flow200=np.load(path1)
        width=int(flow200.shape[0]/18)
        # print(width)
        start=k-width
        a=1
        totalflow = []
        totalflowmic = []
        totalflowmac = []
        mei=1.925
        zui=1.85
        ss=2.10
        # mei=2.4
        # zui=4.2
        # ss=2.10
        totalflow,totalflowmic,totalflowmac=proce2(flow200[0:width,:],1.4,mei,"left_eye",0,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[width:2*width,:],1.4,mei,"left_eye",1,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[2*width:3*width,:],1.4,mei,"left_eye",2,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[3*width:4*width,:],1.4,mei,"left_eye",3,k,a,totalflow,totalflowmic,totalflowmac)

        totalflow,totalflowmic,totalflowmac=proce2(flow200[4*width:5*width,:],1.4,mei,"right_eye",0,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[5*width:6*width,:],1.4,mei,"right_eye",1,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[6*width:7*width,:],1.4,mei,"right_eye",2,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[7*width:8*width,:],1.4,mei,"right_eye",3,k,a,totalflow,totalflowmic,totalflowmac)

        totalflow,totalflowmic,totalflowmac=proce2(flow200[8*width:9*width,:],1.4,zui,"mouth",0,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[9*width:10*width,:],1.4,zui,"mouth",1,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[10*width:11*width,:],1.4,zui,"mouth",2,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[11*width:12*width,:],1.4,zui,"mouth",3,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[12*width:13*width,:],1.4,zui,"mouth",4,k,a,totalflow,totalflowmic,totalflowmac)
        totalflow,totalflowmic,totalflowmac=proce2(flow200[13*width:14*width,:],1.4,zui,"mouth",5,k,a,totalflow,totalflowmic,totalflowmac)

        totalflow, totalflowmic, totalflowmac = proce2(flow200[14*width:15*width,::], 1.4,ss, "nose", 1, k, a, totalflow,totalflowmic, totalflowmac)
        totalflow, totalflowmic, totalflowmac = proce2(flow200[15*width:16*width,::], 1.4,ss, "nose", 2, k, a, totalflow,totalflowmic, totalflowmac)

        totalflow=np.array(nms2(totalflow))  #把所有通道融合起来
        totalflow=np.array(nms2(totalflow))
        totalflowmic_1=np.array(nms2(totalflowmic))
        totalflowmac_1=np.array(nms2(totalflowmac))

        # print(str(k - width) + "--" + str(k) + "all:")
        # print(totalflow)
        totalflow_1=totalflow-(k - width)
        move=100
        for i in range(len(totalflow_1)):
            # if (totalflow_1[i, 0] - (k - hh) < 175):
            if (totalflow_1[i, 0] < 100  and totalflow_1[i, 1] > 100 ):
                if(totalflow_1[i, 1]<150):
                    move=totalflow_1[i, 1]+20
                elif(totalflow_1[i, 0]>50):
                    move=totalflow_1[i, 0]-20
                else:
                    a=min(189,totalflow_1[i, 1])
                    move=a+10

        lable_vio=np.vstack((lable_vio,totalflow))

    lable_video_update = []  # 去除一些太短的片段
    lable_video_update1 = []
    for i in range(len(lable_vio)):

        if (lable_vio[i, 1] - lable_vio[i, 0] >= 12 and lable_vio[i, 1] - lable_vio[i, 0] <= 200):
            lable_video_update.append([lable_vio[i, 0], lable_vio[i, 1]])
    lable_video_update = np.array(nms2(lable_video_update))
    lable_video_update = np.array(nms2(lable_video_update))
    for i in range(len(lable_video_update)):
        if (lable_video_update[i, 1] != 0):
            lable_video_update1.append([lable_video_update[i, 0], lable_video_update[i, 1]])
    lable_video_update1 = np.array(lable_video_update1)
    # print(lable_video_update1)
    return lable_video_update1



