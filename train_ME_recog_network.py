import numpy as np

import torch.utils.data as data
from torchvision import transforms

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
from util.get_ME_feature import getcasmelong,getsammlong,getcasme2

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现



# def parse_args():
#     parser = argparse.ArgumentParser()
#
#     # 数据库地址
#
#     parser.add_argument('--ccac_data_path', type=str,
#                         default="J:/dataset_ME/new_dataset/train_data/train_data",
#                         help='ccac data path.')
#     # A榜数据库 "J:/dataset_ME/new_dataset/test_data_A/test_data_A",  B榜数据库"I:/CCAC/test_data_B/test_data_B"
#
#     parser.add_argument('--ccaclabel_path', type=str, default="ME_label/CCAC2024.csv",
#                         help='CCAC label path.')
#     parser.add_argument('--all_flow_save_path', type=str, default="flow_data/ccac_train_val/",
#                         help='CCAC flow save path.')
#      #A榜标签 "./ME_label/CCAC2024_test_A_withstart.csv",  B榜标签"./ME_label/CCAC2024_test_B.csv"
#
#     parser.add_argument('--flowisexist', type=bool, default=True, help="")
#     # 是否使用数据库
#
#     parser.add_argument('--use_ccac', type=bool, default=True, help='use the ccac dataset.')
#     parser.add_argument('--use_casme2', type=bool, default=True, help='use the casme2 dataset.')
#     parser.add_argument('--use_casmelong', type=bool, default=True, help='use the casmelong dataset.')
#     parser.add_argument('--use_sammlong', type=bool, default=True, help='use the sammlong dataset.')
#
#     parser.add_argument('--casme_first_spotting_result', type=str, default='label/my_casme1.csv', help='')
#     parser.add_argument('--samm_first_spotting_result',  type=str, default='label/my_samm1.csv', help='')
#     parser.add_argument('--casme_label_table', type=str, default="label/CASME2-coding-20190701.csv",
#                         help='')
#
#     parser.add_argument('--calculateflow', type=bool, default=True, help='')
#
#     # 输入是啥
#     parser.add_argument('--use_flow', default=True, type=bool,
#                         help="use flow")
#     parser.add_argument('--use_rois', default=True, type=bool,
#                         help="the numbers of pictures as input")
#     # 训练or测试
#
#     parser.add_argument('--num_classes', type=int, default=4, help='')
#
#     # 训练集和测试集的分割方式
#     parser.add_argument('--use_5_k', default=True, type=bool,
#                         help="recog the ME class")
#     # 模型结构
#     parser.add_argument('--net_name', default="transformer_rois", type=str,
#                         help="a patch consist of 7*7 pixels")  # "res","l2g","alex","res_8pics_flow","transformer_rois"
#
#     # 模型训练
#     parser.add_argument('--device', type=str, default="cuda:0", help='CUDA:0')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
#     parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
#     parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate for sgd.')
#     parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
#     parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers (default: 4)')
#     parser.add_argument('--epochs', type=int, default=10, help='Total training epochs.')
#     parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
#
#     return parser.parse_args()
def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--path_casme', type=str, default="I:/CAS(ME)2/rawpic/", help='casme data path.')
    parser.add_argument('--path_SAMM', type=str, default="I:/SAMM_longvideos/", help='casme data path.')
    parser.add_argument('--path_casme2', type=str, default="I:/dataset_ME/CASME2/CASME2-RAW/sub",
                        help='casme data path.')


    parser.add_argument('--flowisexist', type=bool, default=True, help="")
    # 是否使用数据库

    parser.add_argument('--use_casme2', type=bool, default=True, help='use the casme2 dataset.')
    parser.add_argument('--use_casmelong', type=bool, default=True, help='use the casmelong dataset.')
    parser.add_argument('--use_sammlong', type=bool, default=True, help='use the sammlong dataset.')

    parser.add_argument('--casme_first_spotting_result', type=str, default='label/labeled_spot_result_for_train/my_casme1.csv', help='')
    parser.add_argument('--samm_first_spotting_result',  type=str, default='label/labeled_spot_result_for_train/my_samm1.csv', help='')
    parser.add_argument('--casme_label_table', type=str, default="label/datasetlabel/CASME2-coding-20190701.csv",
                        help='')

    parser.add_argument('--MECLASS_DECTION', type=bool, default=True, help='')
    parser.add_argument('--calculateflow', type=bool, default=True, help='')

    # 输入是啥
    parser.add_argument('--use_flow', default=True, type=bool,
                        help="use flow")
    parser.add_argument('--use_rois', default=True, type=bool,
                        help="the numbers of pictures as input")
    # 训练or测试

    parser.add_argument('--num_classes', type=int, default=4, help='')

    # 训练集和测试集的分割方式
    parser.add_argument('--use_5_k', default=True, type=bool,
                        help="recog the ME class")
    # 模型结构
    parser.add_argument('--net_name', default="transformer_rois", type=str,
                        help="a patch consist of 7*7 pixels")  # "res","l2g","alex","res_8pics_flow","transformer_rois"

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

def getemotion(videoname,start,datasetname="casmelong"):
    if(datasetname=="casmelong"):
        casmelong_label_table = pd.read_csv("label/datasetlabel/CAS(ME)^2code_final(Updated).csv", header=None)
        for index, row in casmelong_label_table.iterrows():
            if(row[10]==videoname and row[2]==start):
                emo=row[8]
                if (emo in "sadness") or (emo in "anger") or (emo in "fear") or (emo in "disgust") or (emo in "helpness") or (emo in "pain"):
                    emotion =3
                elif (emo in "happiness"):
                    emotion = 2
                elif (emo in "surprise"):
                    emotion = 1
                else:
                    print(videoname+str(start))
                    emotion = 4
                return emotion
    if(datasetname=="casme2"):
        emo=start
        if (emo in "sadness") or (emo in "anger") or (emo in "fear") or (emo in "disgust"):
            emotion =3
        elif (emo in "happiness"):
            emotion = 2
        elif (emo in "surprise"):
            emotion = 1
        else:
            print(videoname + str(start))
            emotion = 4
        return emotion
    if (datasetname == "sammlong"):
        print(videoname)
        print(int(start))
        samm_table = pd.read_csv("label/datasetlabel/SAMM_Micro_FACS_Codes_v2.csv", header=None)
        for index, row in samm_table.iterrows():
            if (index >= 16):

                if (row[1].split("_")[0]+"_"+row[1].split("_")[1] == videoname and int(row[3]) == int(start)):
                    print("llllllllllllll")
                    emo = row[9]
                    if (emo in "Sadness") or (emo in "Anger") or (emo in "Fear") or (emo in "Disgust"):
                        emotion = 3
                    elif (emo in "Happiness"):
                        emotion = 2
                    elif (emo in "Surprise"):
                        emotion = 1
                    else:
                        print(videoname + str(start))
                        emotion = 4
                    return emotion


class MEDataSet(data.Dataset):
    def __init__(self, args, phase, train_loso, test_loso):

        casmelong_label_table_all = pd.read_csv(args.casme_first_spotting_result, header=None)
        samm_label_table_all  = pd.read_csv(args.samm_first_spotting_result, header=None)
        casme2_label_table = pd.read_csv(args.casme_label_table, header=None)
        print("train set:",train_loso)
        print("test set:",test_loso)

        if phase == 'train':
            casmelong_label_table = casmelong_label_table_all
            samm_label_table = samm_label_table_all
            for ss in test_loso:
                if ("s" in ss):
                    casmelong_label_table = casmelong_label_table.loc[casmelong_label_table[6] != ss]
                else:
                    samm_label_table      = samm_label_table.loc[samm_label_table[6] != int(ss)]
        else:
            casmelong_label_table = casmelong_label_table_all
            samm_label_table = samm_label_table_all
            for ss in train_loso:
                if("s" in ss  ):
                    casmelong_label_table = casmelong_label_table.loc[casmelong_label_table[6] != ss]
                elif(len(ss)==3  ):
                    samm_label_table      = samm_label_table.loc[samm_label_table[6] != int(ss)]

        self.file_paths_flow_rois=[]
        self.file_paths_on=[]
        self.label_all = []
        self.sub = []
        self.video_names = []
        lenforfeature=25
        if (args.use_sammlong == True):
            print("samm long dataset")
            for index, row in samm_label_table.iterrows():
                if (index >= 0):
                    type = row[5]
                    if (type == "FP" and int(row[4])-int(row[3])<140):
                        emo = 0
                        start = int(row[3])
                        end = int(row[4])
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

                    self.file_paths_flow_rois.append(flow_face)
                    self.sub.append(row[0])
                    self.file_paths_on.append(start)

                    self.label_all.append(emo)
                    self.video_names.append(video_path)
        if (args.use_casme2 == True and phase == 'train'):
            print("casme2 dataset")
            for index, row in casme2_label_table.iterrows():
                if (index >= 0):
                    emotion = row[8]
                    if (emotion in "sadness") or (emotion in "fear") or (emotion in "repression") or (
                            emotion in "disgust"):
                        emo = 3
                    elif (emotion in "happiness"):
                        emo = 2
                    elif (emotion in "surprise"):
                        emo = 1
                    else:
                        emo = 4
                    if (emo == 4):
                        continue

                    start = max(int(row[3]) - 30, 0)
                    end = int(row[5])

                    video_path = args.path_casme2 + row[0] + "/" + row[1] + "/"

                    num=random.randint(1,40)
                    flow_face = getcasme2(video_path, start, end,num)
                    roisnum, n, l = flow_face.shape  # MMNet-main/flow_data/casme3_flow/spNO.160_d_3034.npy
                    if (n < lenforfeature):
                        buling = torch.zeros((roisnum, lenforfeature - n, l))
                        flow_face = torch.cat([flow_face, buling], dim=1)
                    else:
                        flow_face = flow_face[:, 0:lenforfeature, :]
                    self.file_paths_flow_rois.append(flow_face)
                    self.sub.append(row[0])
                    self.file_paths_on.append(start)

                    self.label_all.append(emo)
                    self.video_names.append(video_path)
        if (args.use_casmelong == True):
            print("casmelong dataset")
            # 计算casmelong的
            for index, row in casmelong_label_table.iterrows():
                # print(index)
                if (index >= 0):
                    type=row[5]
                    video_path = args.path_casme + row[0].split("/")[0] + "/" + row[0].split("/")[1] + "/"
                    if(type=="FP"  and int(row[4])-int(row[3])<25):
                        emo=0
                        start=int(row[3])
                        end=int(row[4])

                        flow_face=getcasmelong(video_path,start,end)

                        roisnum,n,l=flow_face.shape#MMNet-main/flow_data/casme3_flow/spNO.160_d_3034.npy
                        if(n<lenforfeature):
                            buling=torch.zeros((roisnum,lenforfeature-n,l))
                            flow_face=torch.cat([flow_face,buling],dim=1)
                        else:
                            flow_face = flow_face[:,0:lenforfeature,:]

                    elif((type=="TP" or type=="FN") and int(row[2])-int(row[1])<20):

                        emo = getemotion(row[0],int(row[1]),"casmelong")
                        start = max(int(row[1])-5,0)
                        end = int(row[2])

                        flow_face = getcasmelong(video_path, start, end)

                        roisnum, n, l = flow_face.shape  # MMNet-main/flow_data/casme3_flow/spNO.160_d_3034.npy
                        if (n < lenforfeature):
                            buling = torch.zeros((roisnum, lenforfeature - n, l))
                            flow_face = torch.cat([flow_face, buling], dim=1)
                        else:
                            flow_face = flow_face[:, 0:lenforfeature, :]

                        if (emo == 4):
                            continue
                    else:
                        continue

                    self.file_paths_flow_rois.append(flow_face)
                    self.sub.append(row[0])
                    self.file_paths_on.append(start)

                    self.label_all.append(emo)
                    self.video_names.append(video_path)
        # from collections import Counter
        # counter = Counter(self.label_all)
        # print(counter)

    def __len__(self):
        return len(self.file_paths_flow_rois)

    def __getitem__(self, idx):
        flow_face=self.file_paths_flow_rois[idx]
        label = self.label_all[idx]
        sub_name = self.sub[idx]
        video_name = self.video_names[idx]
        return  label, flow_face, sub_name, video_name

def contrix(y_true,y_pred,NAME,ISSHOW):

    # 使用sklearn的confusion_matrix函数计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    # 将混淆矩阵转换为DataFrame
    if(ISSHOW is True):
        df_cm = pd.DataFrame(cm, range(len(cm)), range(len(cm)))
        df_cm.columns = ["noME",'surprise','happiness','negative']

        # 使用Seaborn的heatmap函数进行可视化
        sns.heatmap(df_cm, annot=True, fmt='d')
        plt.title(NAME+':Confusion Matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')

        # 显示图形
        plt.show()


def confusionMatrix(gt, pred, show=False):
    # print(gt)
    # print(pred)
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP + 0.000001) / (2 * TP + FP + FN + 0.000001)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall

def recognition_evaluation(final_gt, final_pred):
    label_dict = {'moME': 0, 'surprise': 1, 'surprise': 2,'positive': 0,}
    # label_dict = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''

def run_training_flow(args):
    casmelong_subs = ['s20', 's24', 's33', 's36', 's23', 's27', 's34', 's30', 's38', 's35', 's32', 's31', 's15', 's40', 's25', 's19', 's21', 's16', 's26', 's22', 's37', 's29']
    samm_subs=  ['017', '008', '018', '016', '007', '035', '037', '022', '011', '036', '006', '025', '013', '019', '028', '012', '023', '033', '032', '010', '015', '021', '020', '031', '024', '026', '030', '009', '014', '034']
    casme2_subs = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08', 'sub09', 'sub11', 'sub12',
                   'sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub19', 'sub20', 'sub21', 'sub22', 'sub23', 'sub24',
                   'sub25','sub26']
    datasets = []
    LOSO = []
    sub_fortrain=casme2_subs

    if (args.use_casmelong == True):
        LOSO.extend(casmelong_subs)
        datasets.append("cas(me)2")
    if (args.use_sammlong == True):
        LOSO.extend(samm_subs)
        datasets.append("samm")
    datasets.append("casme2")
    print("use", datasets, "dataset")

    if (args.use_5_k == True):
        kf = KFold(n_splits=5, shuffle=True)
        train_list = []
        test_list = []
        for train_indexs, test_indexs in kf.split(LOSO):
            train_list.append(train_indexs)
            test_list.append(test_indexs)
    y_train_allsub=[]
    y_pred_allsub=[]
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
        train_data.extend(sub_fortrain)
        # for subj in LOSO:
        print("process train set....")
        train_dataset = MEDataSet(args, phase='train', train_loso=train_data, test_loso=test_data)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True)
        print("process test set....")
        val_dataset = MEDataSet(args, phase='test', train_loso=train_data, test_loso=test_data)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 shuffle=False,
                                                 pin_memory=True)
        print('Train set', train_data)
        print('Test set', test_data)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())
        criterion_cls = torch.nn.CrossEntropyLoss(weight=(torch.FloatTensor([1, 3, 9, 9]).to(args.device)))
        ##model initialization
        if (args.net_name == "transformer_rois"):
            net_flow = Transformer_rois()
            print("Transformer_rois")
        params_all = net_flow.parameters()

        if args.optimizer == 'adam':
            optimizer_all = torch.optim.AdamW(params_all, lr=0.0008, weight_decay=0.7)
        else:
            raise ValueError("Optimizer not supported.")

        ##lr_decay
        scheduler_all = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=0.987)
        net_flow = net_flow.to(args.device)


        # 模型训练和测试
        video_list_one_epoch = []
        if(True):
            # 数据分布直方图

            for i in range(1, args.epochs):
                running_loss = 0.0
                iter_cnt = 0
                net_flow.train()

                y_train_one_epoch = []
                y_train_pred_one_epoch = []
                for batch_i, (label_all, flow_faces, sub_sets,  video_name) in enumerate(train_loader):

                    iter_cnt += 1
                    # print(iter_cnt)
                    label_all = label_all.to(args.device)
                    ##train

                    flow_faces = flow_faces.to(args.device)
                    flow_faces = flow_faces.float()
                    ALL = net_flow(flow_faces)

                    loss_cls = criterion_cls(ALL[:, 0:args.num_classes], label_all)
                    loss_all =  loss_cls

                    optimizer_all.zero_grad()
                    loss_all.backward()
                    optimizer_all.step()
                    running_loss += loss_all

                    _, cls_predicts = torch.max(ALL[:, 0:args.num_classes], 1)
                    y_train_one_epoch.extend(label_all.tolist())
                    y_train_pred_one_epoch.extend(cls_predicts.tolist())

                    # print("计算完一个batch", iter_cnt)
                ## lr decay
                if i <= 50:
                    scheduler_all.step()
                running_loss = running_loss / iter_cnt


                UF1_sub_one_epoch, UAR_sub_one_epoch = recognition_evaluation(y_train_one_epoch,
                                                                              y_train_pred_one_epoch)
                print('[Epoch %d] Training CLS_UAR: %.4f. Loss: %.3f' % (i, UAR_sub_one_epoch, running_loss))
                # 绘制x,y随epoch变化图像

                ##val
                # save_params_list=["model_au_1epoch8UF10.5754728086652946.pt","model_au_2epoch9UF10.6392157038533246.pt","model_au_3epoch9UF10.5825491040435234.pt","model_au_4epoch8UF10.506172850553269.pt","model_au_5epoch8UF10.47214286389407584.pt"]
                # args.save_model_params_test=save_params_list[test_time-1]
                # net_flow = Transformer_rois()
                # print("模型名称是Transformer_rois")
                # net_flow.load_state_dict(torch.load(args.save_model_params_test))
                # net_flow = net_flow.to(args.device)

                net_flow.eval()
                with torch.no_grad():
                    y_test_one_epoch = []
                    y_test_pred_one_epoch = []
                    sub_one_epoch = []

                    running_loss = 0.0
                    iter_cnt = 0
                    for batch_i, (label_all, flow_faces, sub_sets,  video_name) in enumerate(val_loader):
                        label_all = label_all.to(args.device)

                        flow_faces = flow_faces.to(args.device)
                        flow_faces = flow_faces.float()
                        ALL = net_flow(flow_faces)

                        # result

                        loss_cls = criterion_cls(ALL[:, 0:args.num_classes], label_all)
                        _, cls_predicts = torch.max(ALL[:, 0:args.num_classes], 1)
                        y_test_one_epoch.extend(label_all.tolist())
                        y_test_pred_one_epoch.extend(cls_predicts.tolist())

                        loss_all = loss_cls
                        running_loss += loss_all
                        iter_cnt += 1

                        sub_one_epoch.extend(sub_sets)
                        video_list_one_epoch.extend(video_name)
                    running_loss = running_loss / iter_cnt


                    contrix(y_test_one_epoch,y_test_pred_one_epoch, "l",False)
                    UF1_sub_one_epoch, UAR_sub_one_epoch = recognition_evaluation(y_test_one_epoch,
                                                                                  y_test_pred_one_epoch)
                    print("[Epoch %d] val Loss:%.3f, UF1-score:%.3f, UAR:%.3f" % (
                        i, running_loss, UF1_sub_one_epoch, UAR_sub_one_epoch))
                if os.path.exists("./save_params") is False:
                    os.makedirs("./save_params")
                torch.save(net_flow.state_dict(), 'save_params/model_' + str(test_time) + 'epoch' + str(i)+'UF1'+str(UF1_sub_one_epoch) + '.pt')

            y_train_allsub.extend(y_test_one_epoch)
            y_pred_allsub.extend(y_test_pred_one_epoch)
    contrix(y_train_allsub, y_pred_allsub, "l",True)
    UF1_sub_one_epoch, UAR_sub_one_epoch = recognition_evaluation(y_train_allsub,
                                                                  y_pred_allsub)
    print("[Epoch %d] val Loss:%.3f, UF1-score:%.3f, UAR:%.3f" % (
        i, running_loss, UF1_sub_one_epoch, UAR_sub_one_epoch))
if __name__ == "__main__":
    RANDOM_SEED = 39  # any random number
    set_seed(RANDOM_SEED)
    args = parse_args()
    run_training_flow(args)



