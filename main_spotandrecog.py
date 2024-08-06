from test_ME_recog import recogME_testset
import spot_util_samm
import spot_util_casme

from time import *
import numpy as np
import  torch
import argparse

torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
import sys
import os
sys.path.append('./model')
import csv

import random
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
def parse_args():
    parser = argparse.ArgumentParser()

    # path of dataset
    parser.add_argument('--path_casme', type=str,default="I:/CAS(ME)2/rawpic/",help='casme data path.')
    parser.add_argument('--path_SAMM', type=str,default="I:/SAMM_longvideos/",help='casme data path.')
    parser.add_argument('--path_casme2', type=str,default="I:/dataset_ME/CASME2/CASME2-RAW/sub",help='casme data path.')


    # path of spotresult
    parser.add_argument('--needspot', type=bool,default=False,help='if need spot,set it true, and then the spot result will write in label/casme_spot_result_test.csv and label/samm_spot_result_test.csv')
    parser.add_argument('--casme_spot_result_test', type=str,default="label/spot_result_test/casme_spot_result_test.csv",help='casme_spot_result.')
    parser.add_argument('--samm_spot_result_test', type=str,default="label/spot_result_test/samm_spot_result_test.csv",help='samm_spot_result')

    # path of spot and rocog result
    parser.add_argument('--casme_spot_recog_result', type=str, default="label/spot_recog_result/cas_pred.csv",
                        help='casme_spot_recog_result')
    parser.add_argument('--samm_spot_recog_result', type=str, default="label/spot_recog_result/samm_pred.csv",
                        help='samm_spot_recog_result')

    #if the flow feature of casme is exist.
    parser.add_argument('--casme_feature_exist', type=bool, default=True, help="")
    parser.add_argument('--casme_feature_path', type=str, default="J:/test_casme1_face8/test_casme1_face8/", help="")

    #使用哪些数据集训练
    parser.add_argument('--use_casme2', type=bool, default=True, help='use the casme2 dataset to train')
    parser.add_argument('--use_casmelong', type=bool, default=True, help='use the casmelong dataset.')
    parser.add_argument('--use_sammlong', type=bool, default=True, help='use the sammlong dataset.')

    #
    parser.add_argument('--casme_first_spotting_result', type=str, default='label/labeled_spot_result_for_train/my_casme1.csv', help='')
    parser.add_argument('--samm_first_spotting_result',  type=str, default='label/labeled_spot_result_for_train/my_samm1.csv', help='')
    parser.add_argument('--casme_label_table', type=str, default="label/datasetlabel/CASME2-coding-20190701.csv",
                        help='')
    parser.add_argument('--casme2_label_table', type=str, default="label/datasetlabel/CAS(ME)^2code_final(Updated).csv",
                        help='')

    parser.add_argument('--calculateflow', type=bool, default=True, help='')

    # 输入是啥
    parser.add_argument('--use_flow', default=True, type=bool,
                        help="use flow")
    parser.add_argument('--use_rois', default=True, type=bool,
                        help="the numbers of pictures as input")
    # 训练or测试



    # 训练集和测试集的分割方式
    parser.add_argument('--use_5_k', default=True, type=bool,
                        help="recog the ME class")
    # 模型结构
    parser.add_argument('--net_name', default="transformer_rois", type=str,
                        help="a patch consist of 7*7 pixels")  # "res","l2g","alex","res_8pics_flow","transformer_rois"
    parser.add_argument('--num_classes', type=int, default=4, help='')
    # 模型训练
    parser.add_argument('--device', type=str, default="cuda:0", help='CUDA:0')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=9, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')

    return parser.parse_args()

def multivideo_SAMM(SAMM_data_path,samm_spotresult):
    fileList = os.listdir(SAMM_data_path)
    for vio in fileList:
        print("=================================================")
        print("vio:" + vio)
        if(True):
            path5=vio
            #SAMM
            spot_result_one_video = spot_util_samm.draw_roiline19(SAMM_data_path , path5 , 6, -4,7)  #18是直接使用光流计算，19是全部，20是去掉全局移动
            spot_result_one_video=spot_result_one_video*7
            print("result:")
            print(spot_result_one_video)
            for j in range(len(spot_result_one_video)):
                start2 = spot_result_one_video[j, 0] + 1  # 预测的标签
                end2 = spot_result_one_video[j, 1] + 1
                with open(samm_spotresult, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([path5, start2, end2,path5.split("_")[0]])
def multivideo_CASME(casme_data_path,casme_spotresult):
    fileList = os.listdir(casme_data_path)
    k=0
    for sub in fileList:
        print("=================================================")
        print("sub"+sub)
        starttime=time()
        k=k+1
        if(k>=1):  #从哪个被试开始
            path_sub=casme_data_path + sub+"/"
            fileList_vio = os.listdir(path_sub)
            for vio in fileList_vio:
                print("=====================")
                print("spot video" + vio)
                path5=sub+"/"+vio
                # CAS(ME)2
                if os.path.exists(args.casme_feature_path) is False:
                    print("upzip the casme feature, put these in the args.casme_feature_path")
                else:
                    pp = spot_util_casme.draw_roiline18(casme_data_path , path5 , 4, -4,1,args.casme_feature_path)
                    print("result:")
                    print(pp)
                    for j in range(len(pp)):
                        start2 = pp[j, 0] + 1
                        end2 = pp[j, 1] + 1
                        with open(casme_spotresult, "a", newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([path5, start2, end2,path5.split("/")[0]])
        endtime=time()
        print("time of this subjuect is {}".format(endtime-starttime))
if __name__ == "__main__":
    RANDOM_SEED = 39  # any random number
    set_seed(RANDOM_SEED)
    args = parse_args()
    #spot ME
    if(args.needspot):
        print("spot the CAS(ME)2 dataset, and save the spot results in the file named args.casme_spot_result_test")
        # multivideo_CASME(args.path_casme,args.casme_spot_result_test)
        print("spot the SAMM long dataset, and save the spot results in the file named args.samm_spot_result_test")
        multivideo_SAMM(args.path_SAMM,args.samm_spot_result_test)

    print("We use five-fold cross-validation to split the training set and test set.")
    print("We use a fixed random seed to ensure the reproducibility of the experiments.")
    print("To avoid overfitting, we train only 8 epochs for every fold.")
    print("make sure the result csv (label/spot_recog_result/cas_pred.csv and label/spot_recog_result/samm_pred) have only head.")
    #Recognize micro-expressions (ME)
    recogME_testset(args)
