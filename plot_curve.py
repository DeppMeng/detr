import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
from os import listdir
from os.path import isfile, join

# cmap = plt.get_cmap("rainbow")
cmap = plt.get_cmap("Dark2")

pattern_seg_epoch = re.compile("Epoch: \[([0-9]*)/")
pattern_seg_miou = re.compile("Loss: [0-9]*\.?[0-9]*, MeanIU:  ([0-9]*\.?[0-9]*), Best_mIoU:  [0-9]*\.?[0-9]*")
pattern_seg_train_loss = re.compile("lr: [0-9]*\.?[0-9]*, Loss: ([0-9]*\.?[0-9]*)")

pattern_val_ap = re.compile("\| pose_sma\.\.\. \| ([0-9]*\.?[0-9]*) \|")
pattern_train_loss = re.compile("Data [0-9]*\.?[0-9]*s? \([0-9]*\.?[0-9]*s?\)\sLoss [0-9]*\.?[0-9]* \(([0-9]*\.?[0-9]*)\)")


pattern_cls_train_inst_acc_256 = re.compile("\[1847\/1848\]  eta: [0-9]:[0-9][0-9]:[0-9][0-9]  lr: [0-9]\.[0-9]+  class_error: [0-9]+\.[0-9]+  loss: [0-9]*\.?[0-9]* \(([0-9]*\.?[0-9]*)\)")

# [1847/1848]  eta: 0:00:00  lr: 0.000100  class_error: 95.54  loss: 22.7429 (25.9595)

def get_inst_loss(file):
    match_name = []
    match_list = []
    pair_list = []

    pattern = pattern_cls_train_inst_acc_256

    # pattern_pair = pattern_cls_train_inst_pair_acc_256

    for i, line in enumerate(open(file, encoding='UTF-8')):
        for match_name in re.finditer(pattern, line):
            match_list.append(float(match_name.group(1)))

    return match_list


# def get_pair_loss(file, bs, w, w_f, anneal_type):
#     match_name = []
#     pair_list = []

#     pattern_pair = pattern_cls_train_inst_pair_acc_256

#     for i, line in enumerate(open(file, encoding='UTF-8')):
#         for match_name in re.finditer(pattern_pair, line):
#             pair_list.append(float(match_name.group(1)))
#     print(pair_list[-1])
#     return pair_list


def plot_inst_loss(miou_list, name_list):
    max_epoch = max([len(l) for l in miou_list])
    epochs = [i for i in range(max_epoch)]
    x_ticks = [i*10 for i in range(int(max_epoch / 10))]
    y_ticks = np.linspace(7, 20, np.round((20 - 7) / 4) + 1, endpoint=True)
    p_list = []
    for i in range(len(miou_list)):
        p_list.append(plt.plot([j for j in range(len(miou_list[i]))], miou_list[i], color=cmap(i)))
        # p_list.append(plt.plot([j for j in range(len(miou_list[i]))], [j for j in range(len(miou_list[i]))], color=cmap(i)))
    plt.xlabel('epoch')
    plt.ylabel('train-loss')
    plt.legend([p[0] for p in p_list], name_list)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    # plt.title('Pretrain on ImageNet-1k')
    plt.ylim([7, 20])
    plt.grid(linestyle='--')
    plt.show()

# def plot_pair_loss(miou_list, name_list):
#     max_epoch = max([len(l) for l in miou_list])
#     epochs = [i for i in range(max_epoch)]
#     x_ticks = [i*10 for i in range(int(max_epoch / 10))]
#     # y_ticks = np.linspace(0.0, 0.08, np.round((0.08 - 0.0) / 4) + 1, endpoint=True)
#     p_list = []
#     for i in range(len(miou_list)):
#         p_list.append(plt.plot([j for j in range(len(miou_list[i]))], miou_list[i], color=cmap(i)))
#         # p_list.append(plt.plot([j for j in range(len(miou_list[i]))], [j for j in range(len(miou_list[i]))], color=cmap(i)))
#     plt.xlabel('epoch')
#     plt.ylabel('pair-loss')
#     plt.legend([p[0] for p in p_list], name_list)
#     plt.xticks(x_ticks)
#     # plt.yticks(y_ticks)
#     plt.title('Pretrain on ImageNet-1k')
#     plt.ylim([0.0, 0.08])
#     plt.grid(linestyle='--')
#     plt.show()

if __name__ == '__main__':
    import sys

    plot_type = 'inst-loss'
    # plot_type = 'pair-loss'


    name_list = [
        '$learned$',
        '$SineV4$',
        '$SineV4+Trans$',
        ]

    exp_list = [
        'logs/log_learned_epoch150_4nodes_train.txt',
        'logs/log_v4_epoch150_4nodes_train.txt',
        'logs/log_v4_trans_epoch150_4nodes_train.txt',
    ]
    
    if plot_type == 'inst-loss':
        train_inst_loss_list = []
        for i, exp in enumerate(exp_list):
            # if i == 0:
            #     tmp_acc_folder.extend([0 for i in range(33)])
            # files_paths = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) and '.txt' in f]
            # for file_path in files_paths:
            tmp_inst_loss_file = get_inst_loss(exp_list[i])
            train_inst_loss_list.append(tmp_inst_loss_file)
                
        plot_inst_loss(train_inst_loss_list, name_list)
        
    # elif plot_type == 'pair-loss':
    #     train_pair_loss_list = []
    #     for i, exp in enumerate(exp_list):
    #         # if i == 0:
    #         #     tmp_acc_folder.extend([0 for i in range(33)])
    #         # files_paths = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) and '.txt' in f]
    #         # for file_path in files_paths:
    #         tmp_pair_loss_file = get_pair_loss(exp_list[i], bs_list[i], w_list[i], wf_list[i], anneal_list[i])
    #         train_pair_loss_list.append(tmp_pair_loss_file)
                
        # plot_pair_loss(train_pair_loss_list, name_list)