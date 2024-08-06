import numpy as np

import torch.utils.data as data

import pandas as pd
import  torch

from sklearn.model_selection import KFold
import argparse
# from tensorboardX import SummaryWriter
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
import sys
import os
sys.path.append('./model')
from collections import Counter
from model.transformer_block import Transformer_rois
import csv

import random
from util.get_ME_feature import getcasmelong,getsammlong


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
    parser.add_argument('--path_casme', type=str, default="I:/CAS(ME)2/rawpic/", help='casme data path.')
    parser.add_argument('--path_SAMM', type=str, default="I:/SAMM_longvideos/", help='casme data path.')

    # path of spotresult
    parser.add_argument('--casme_spot_result_test', type=str, default="label/spot_result_test/casme_spot_result_test.csv",
                        help='casme data path.')
    parser.add_argument('--samm_spot_result_test', type=str, default="label/spot_result_test/samm_spot_result_test.csv",
                        help='casme data path.')

    # path of spot and rocog result
    parser.add_argument('--casme_spot_recog_result', type=str, default="label/spot_recog_result/cas_pred.csv",
                        help='casme_spot_recog_result')
    parser.add_argument('--samm_spot_recog_result', type=str, default="label/spot_recog_result/samm_pred.csv",
                        help='samm_spot_recog_result')

    # if the flow feature of casme is exist.
    parser.add_argument('--casme_feature_exist', type=bool, default=True, help="")
    parser.add_argument('--casme_feature_path', type=str, default="J:/test_casme1_face8/test_casme1_face8/", help="")

    # if train the network,or use the exist params
    parser.add_argument('--train_network', type=bool, default=False, help="")

    # 使用哪些数据集训练
    parser.add_argument('--use_casme2', type=bool, default=True, help='use the casme2 dataset to train')
    parser.add_argument('--use_casmelong', type=bool, default=True, help='use the casmelong dataset.')
    parser.add_argument('--use_sammlong', type=bool, default=True, help='use the sammlong dataset.')

    #
    parser.add_argument('--casme_first_spotting_result', type=str, default='label/labeled_spot_result_for_train/my_casme1.csv',
                        help='')
    parser.add_argument('--samm_first_spotting_result', type=str, default='label/labeled_spot_result_for_train/my_samm1.csv',
                        help='')
    parser.add_argument('--casme_label_table', type=str, default="label/datasetlabel/CASME2-coding-20190701.csv",
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
    parser.add_argument('--epochs', type=int, default=10, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')


    return parser.parse_args()

class MEDataSet_spotandreg(data.Dataset):
    def __init__(self, args,  train_loso):

        casmelong_label_table_all = pd.read_csv(args.casme_spot_result_test, header=None)
        samm_label_table_all  = pd.read_csv(args.samm_spot_result_test, header=None)

        casmelong_label_table = casmelong_label_table_all
        samm_label_table = samm_label_table_all
        for ss in train_loso:
            if("s" in ss  ):
                casmelong_label_table = casmelong_label_table.loc[casmelong_label_table[3] != ss]
            elif(len(ss)==3  ):
                samm_label_table      = samm_label_table.loc[samm_label_table[3] != int(ss)]

        self.file_paths_on=[]
        self.file_paths_flow_rois = []
        self.label_all = []
        self.sub = []
        self.video_names = []
        self.spot_start = []
        self.spot_end = []

        lenforfeature=25
        if (args.use_sammlong == True):
            for index, row in samm_label_table.iterrows():
                if (index >= 0 ):

                    if (int(row[2])-int(row[1])<160):
                        emo = 0
                        start = int(row[1])
                        end = int(row[2])
                    else:
                        continue

                    video_path = args.path_SAMM + row[0] + "/"
                    flow_face = getsammlong(video_path, start, end)
                    roisnum, n, l = flow_face.shape  # MMNet-main/flow_data/casme3_flow/spNO.160_d_3034.npy
                    if (n < lenforfeature):
                        buling = torch.zeros((roisnum, lenforfeature - n, l))
                        flow_face = torch.cat([flow_face, buling], dim=1)
                    else:
                        flow_face = flow_face[:, 0:lenforfeature, :]
                    # print(video_path)

                    self.file_paths_flow_rois.append(flow_face)
                    self.sub.append(row[0])
                    self.file_paths_on.append(start)
                    self.label_all.append(emo)
                    self.video_names.append(video_path)
                    self.spot_start.append(start)
                    self.spot_end.append(end)

        if (args.use_casmelong == True):

            # 计算casmelong的
            for index, row in casmelong_label_table.iterrows():
                # print(index)
                if (index >= 0 ):

                    video_path = args.path_casme + row[0].split("/")[0] + "/" + row[0].split("/")[1] + "/"
                    if (int(row[2])-int(row[1])<25):
                        start = max(int(row[1]) - 5, 0)
                        end = int(row[2])
                        emo = 0
                    else:
                        continue


                    flow_face=getcasmelong(video_path,start,end)

                    roisnum,n,l=flow_face.shape#MMNet-main/flow_data/casme3_flow/spNO.160_d_3034.npy
                    if(n<lenforfeature):
                        buling=torch.zeros((roisnum,lenforfeature-n,l))
                        flow_face=torch.cat([flow_face,buling],dim=1)
                    else:
                        flow_face = flow_face[:,0:lenforfeature,:]

                    # onset_path=args.ccac_data_path + "/" + str(row[1]) + "/" + str(row[2]).zfill(5) + ".png"
                    # flow_path_rois = flow_savepath + "ccac_train_val/" + str(row[0]) + "/" + str(row[1]) + "/" + "flow_rois.npy"
                    # print(video_path)
                    self.file_paths_flow_rois.append(flow_face)
                    self.sub.append(row[0])
                    self.file_paths_on.append(start)

                    self.label_all.append(emo)
                    self.video_names.append(video_path)

                    self.spot_start.append(start)
                    self.spot_end.append(end)

        # counter = Counter(self.label_all)
        # print(counter)

    def __len__(self):
        return len(self.file_paths_flow_rois)

    def __getitem__(self, idx):
        ##sampling strategy for training set
        flow_face=self.file_paths_flow_rois[idx]
        label = self.label_all[idx]
        sub_name = self.sub[idx]
        video_name = self.video_names[idx]
        start=self.spot_start[idx]
        end=self.spot_end[idx]
        return  label, flow_face, sub_name, video_name,start,end

def recogME_testset(args):

    casmelong_subs = ['s20', 's24', 's33', 's36', 's23', 's27', 's34', 's30', 's38', 's35', 's32', 's31', 's15', 's40',
                      's25', 's19', 's21', 's16', 's26', 's22', 's37', 's29']
    samm_subs = ['017', '008', '018', '016', '007', '035', '037', '022', '011', '036', '006', '025', '013', '019',
                 '028', '012', '023', '033', '032', '010', '015', '021', '020', '031', '024', '026', '030', '009',
                 '014', '034']

    datasets = []
    LOSO = []

    if (args.use_casmelong == True):
        LOSO.extend(casmelong_subs)
        datasets.append("cas(me)2")
    if (args.use_sammlong == True):
        LOSO.extend(samm_subs)
        datasets.append("samm")
    print("test", datasets, "dataset")

    if (args.use_5_k == True):
        kf = KFold(n_splits=5, shuffle=True)
        train_list = []
        test_list = []
        for train_indexs, test_indexs in kf.split(LOSO):
            train_list.append(train_indexs)
            test_list.append(test_indexs)
    y_train_allsub = []
    y_pred_allsub = []
    sub_allsub=[]
    videoname_allsub = []
    start_allsub=[]
    end_allsub=[]
    test_time = 0
    for train_indexs, test_indexs in zip(train_list, test_list):  # 调用split方法切分数据
        test_time += 1
        # if(test_time>1):
        #     break
        if (args.use_5_k == True):
            # 构建数据
            test_data = []
            for test_ind in test_indexs:
                test_data.append(LOSO[test_ind])
            train_data = []
            for train_ind in train_indexs:
                train_data.append(LOSO[train_ind])

        val_dataset = MEDataSet_spotandreg(args,  train_loso=train_data)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 shuffle=False,
                                                 pin_memory=True)
        ##val
        save_params_list = ["model_au_1epoch8UF10.6378737726018464.pt", "model_au_2epoch8UF10.5139535055354237.pt",
                            "model_au_3epoch8UF10.4735202880633234.pt", "model_au_4epoch8UF10.5126984378420237.pt",
                            "model_au_5epoch8UF10.5891346145502192.pt"]
        print("===================================")
        print("the 5Fold: ",test_time)
        print('Test sujects consist: ', test_data)
        print('Number of spotted intervals is: ', val_dataset.__len__())
        print("The model params is: ", save_params_list[test_time-1])

        if (True):
            save_model_params_test = "save_params/" + save_params_list[test_time - 1]
            net_flow = Transformer_rois()
            print("name of model is Transformer_rois")
            net_flow.load_state_dict(torch.load(save_model_params_test))
            net_flow = net_flow.to(args.device)

            net_flow.eval()
            with torch.no_grad():
                y_test_one_epoch = []
                y_test_pred_one_epoch = []
                sub_one_epoch = []
                start_one_epoch=[]
                end_one_epoch=[]
                video_list_one_epoch = []

                iter_cnt = 0
                for batch_i, (label_all, flow_faces, sub_sets, video_name,start,end) in enumerate(val_loader):
                    label_all = label_all.to(args.device)
                    if (args.use_flow):
                        flow_faces = flow_faces.to(args.device)
                        flow_faces = flow_faces.float()
                        ALL = net_flow(flow_faces)  #the result


                    _, cls_predicts = torch.max(ALL[:, 0:args.num_classes], 1)

                    y_test_one_epoch.extend(label_all.tolist())
                    y_test_pred_one_epoch.extend(cls_predicts.tolist())
                    sub_one_epoch.extend(sub_sets)
                    video_list_one_epoch.extend(video_name)
                    start_one_epoch.extend(start)
                    end_one_epoch.extend(end)
                    iter_cnt += 1

                sub_allsub.extend(sub_one_epoch)
                y_train_allsub.extend(y_test_one_epoch)
                y_pred_allsub.extend(y_test_pred_one_epoch)
                videoname_allsub.extend(video_list_one_epoch)
                start_allsub.extend(start_one_epoch)
                end_allsub.extend(end_one_epoch)
    #write the test result to csv file

    d= ['s15', 's16', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's29', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's40']
    emotionlist=["no","surprise","positive","negative"]
    for i in range(len(sub_allsub)):
        # print(sub_allsub[i],y_pred_allsub[i],videoname_allsub[i],start_allsub[i],end_allsub[i])
        # cas(me)2
        if(sub_allsub[i][0]=="s"):
            subname=sub_allsub[i].split("/")[0]
            subindex=d.index(subname)+1
            startindex=int(start_allsub[i])
            endindex=int(end_allsub[i])
            label=y_pred_allsub[i]

            if(int(label)==0):
                # print("排除无关")
                continue
            else:
                emotion = emotionlist[int(label)]
            casme2label_table = pd.read_csv(args.casme2_label_table, header=None)
            for index, row in casme2label_table.iterrows():
                if (row[10] ==sub_allsub[i]):
                    video=row[1].split("_")[0]
            with open(args.casme_spot_recog_result, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([subindex,video, startindex,endindex,emotion])
                print(subindex,video, startindex,endindex,emotion)
        # samm
        else:

            subname = int(sub_allsub[i].split("_")[0])
            video=sub_allsub[i]

            startindex = int(start_allsub[i])
            endindex = int(end_allsub[i])

            label = y_pred_allsub[i]
            if (int(label) == 0):
                # print("排除无关")
                continue
            else:
                emotion = emotionlist[int(label)]
            with open(args.samm_spot_recog_result, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([subname, video, startindex, endindex, emotion])
                print(subname, video, startindex, endindex, emotion)

if __name__ == "__main__":
    RANDOM_SEED = 39  # any random number
    set_seed(RANDOM_SEED)
    args = parse_args()
    # if args.flowisexist is False:
    #     get_train_feature(args.ccac_data_path,args.ccaclabel_path,args.all_flow_save_path)
    # run_training_flow(args)
    # onlytest(args)
    recogME_testset(args)

