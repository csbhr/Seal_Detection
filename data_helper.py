import numpy as np
import os
import shutil
import cv2
import random


def copy_files(origin_file_paths, dest_file_paths):
    '''
    copy files from origin path to dest path
    :param origin_file_paths: a list, origin files' full path
    :param dest_file_paths: a list, dest files' full path
    :return:
    '''
    for ori, dest in zip(origin_file_paths, dest_file_paths):
        shutil.copyfile(ori, dest)
        print("-- copy file {} ---".format(os.path.basename(ori)))


# # 把fake从硬盘中放入目录中
# ori_root = "/media/csbhr/LittleBear/study_source/WorkSpace/python/project_data/Seal_Detection/fake"
# dest_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/before_extend/all/wrong_with_position"
#
# fname_list = os.listdir(ori_root)
#
# ori_path_list = [os.path.join(ori_root, fn) for fn in fname_list]
# dest_path_list = []
#
# for fn in fname_list:
#     basename, position = fn.split("_")[0], fn.split("_")[2].split(".")[0]
#     dest_path_list.append(os.path.join(dest_root, "{}_{}.jpg".format(basename, position)))
#
# copy_files(ori_path_list, dest_path_list)


# # 从Seal_Detection转移到CNN_LSTM_CTC_Tensorflow
# ori_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/before_extend/all"
# ori_origin_root = os.path.join(ori_root, "origin")
# ori_right_root = os.path.join(ori_root, "right")
# ori_wrong_root = os.path.join(ori_root, "wrong_with_position")
# dest_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/CNN_LSTM_CTC_Tensorflow/unlabeled/before_extend/all"
# dest_right_root = os.path.join(dest_root, "right")
# dest_wrong_root = os.path.join(dest_root, "wrong")
#
# ori_right_path_list = []
# ori_wrong_path_list = []
# dest_right_path_list = []
# dest_wrong_path_list = []
#
# index = 0
#
# fname_list = os.listdir(ori_origin_root)
# for fn in fname_list:
#     index += 1
#     ori_right_path_list.append(os.path.join(ori_origin_root, fn))
#     dest_right_path_list.append(os.path.join(dest_right_root, "{}_{}".format(index, fn)))
# fname_list = os.listdir(ori_right_root)
# for fn in fname_list:
#     index += 1
#     ori_right_path_list.append(os.path.join(ori_right_root, fn))
#     dest_right_path_list.append(os.path.join(dest_right_root, "{}_{}".format(index, fn)))
# fname_list = os.listdir(ori_wrong_root)
# for fn in fname_list:
#     index += 1
#     ori_wrong_path_list.append(os.path.join(ori_wrong_root, fn))
#     dest_wrong_path_list.append(os.path.join(dest_wrong_root, "{}_{}".format(index, fn)))
#
# copy_files(ori_right_path_list, dest_right_path_list)
# copy_files(ori_wrong_path_list, dest_wrong_path_list)


# # 分割train和val
# ori_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/CNN_LSTM_CTC_Tensorflow/unlabeled/before_extend/all"
# ori_right_root = os.path.join(ori_root, "right")
# ori_wrong_root = os.path.join(ori_root, "wrong")
# dest_train_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/CNN_LSTM_CTC_Tensorflow/unlabeled/before_extend/train"
# dest_train_right_root = os.path.join(dest_train_root, "right")
# dest_train_wrong_root = os.path.join(dest_train_root, "wrong")
# dest_val_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/CNN_LSTM_CTC_Tensorflow/unlabeled/before_extend/val"
# dest_val_right_root = os.path.join(dest_val_root, "right")
# dest_val_wrong_root = os.path.join(dest_val_root, "wrong")
#
# train_dict = {
#     "ori_right": [],
#     "dest_right": [],
#     "ori_wrong": [],
#     "dest_wrong": []
# }
# val_dict = {
#     "ori_right": [],
#     "dest_right": [],
#     "ori_wrong": [],
#     "dest_wrong": []
# }
#
# fname_list = os.listdir(ori_right_root)
# val_thre = int(len(fname_list) * 0.8)
# for fn in fname_list[:val_thre]:
#     train_dict["ori_right"].append(os.path.join(ori_right_root, fn))
#     train_dict["dest_right"].append(os.path.join(dest_train_right_root, fn))
# for fn in fname_list[val_thre:]:
#     val_dict["ori_right"].append(os.path.join(ori_right_root, fn))
#     val_dict["dest_right"].append(os.path.join(dest_val_right_root, fn))
#
# fname_list = os.listdir(ori_wrong_root)
# val_thre = int(len(fname_list) * 0.8)
# for fn in fname_list[:val_thre]:
#     train_dict["ori_wrong"].append(os.path.join(ori_wrong_root, fn))
#     train_dict["dest_wrong"].append(os.path.join(dest_train_wrong_root, fn))
# for fn in fname_list[val_thre:]:
#     val_dict["ori_wrong"].append(os.path.join(ori_wrong_root, fn))
#     val_dict["dest_wrong"].append(os.path.join(dest_val_wrong_root, fn))
#
# copy_files(train_dict["ori_right"], train_dict["dest_right"])
# copy_files(train_dict["ori_wrong"], train_dict["dest_wrong"])
# copy_files(val_dict["ori_right"], val_dict["dest_right"])
# copy_files(val_dict["ori_wrong"], val_dict["dest_wrong"])


# # 拓展train数据集
# def extend_img(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     (h, w) = img.shape[:2]
#     center = (w // 2, h // 2)
#     M_1 = cv2.getRotationMatrix2D(center, 3, 1.0)
#     rotated_1 = cv2.warpAffine(img, M_1, (w, h))
#     M_2 = cv2.getRotationMatrix2D(center, -3, 1.0)
#     rotated_2 = cv2.warpAffine(img, M_2, (w, h))
#     flip_0 = cv2.flip(img, 0)
#     flip_1 = cv2.flip(img, 1)
#     flip_2 = cv2.flip(img, -1)
#     return img, rotated_1, rotated_2, flip_0, flip_1, flip_2
#
# ori_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/unlabeled/before_extend/train"
# ori_right_root = os.path.join(ori_root, "right")
# ori_wrong_root = os.path.join(ori_root, "wrong")
# dest_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/unlabeled/after_extend/train"
# dest_right_root = os.path.join(dest_root, "right")
# dest_wrong_root = os.path.join(dest_root, "wrong")
#
# fname_list = os.listdir(ori_right_root)
# for fn in fname_list:
#     img, rotated_1, rotated_2, flip_0, flip_1, flip_2 = extend_img(os.path.join(ori_right_root, fn))
#     ind, na = int(fn.split("_")[0]), fn.split("_")[1]
#     cv2.imwrite(os.path.join(dest_right_root, "{}_{}".format(10000 + ind, na)), img)
#     cv2.imwrite(os.path.join(dest_right_root, "{}_{}".format(20000 + ind, na)), rotated_1)
#     cv2.imwrite(os.path.join(dest_right_root, "{}_{}".format(30000 + ind, na)), rotated_2)
#     cv2.imwrite(os.path.join(dest_right_root, "{}_{}".format(40000 + ind, na)), flip_0)
#     cv2.imwrite(os.path.join(dest_right_root, "{}_{}".format(50000 + ind, na)), flip_1)
#     cv2.imwrite(os.path.join(dest_right_root, "{}_{}".format(60000 + ind, na)), flip_2)
#
# fname_list = os.listdir(ori_wrong_root)
# for fn in fname_list:
#     img, rotated_1, rotated_2, flip_0, flip_1, flip_2 = extend_img(os.path.join(ori_wrong_root, fn))
#     ind, la, pos = int(fn.split("_")[0]), fn.split("_")[1], int(fn.split("_")[2].split(".")[0])
#     cv2.imwrite(os.path.join(dest_wrong_root, "{}_{}_{}.jpg".format(10000 + ind, la, pos)), img)
#     cv2.imwrite(os.path.join(dest_wrong_root, "{}_{}_{}.jpg".format(20000 + ind, la, pos)), rotated_1)
#     cv2.imwrite(os.path.join(dest_wrong_root, "{}_{}_{}.jpg".format(30000 + ind, la, pos)), rotated_2)
#     cv2.imwrite(os.path.join(dest_wrong_root, "{}_{}_{}.jpg".format(40000 + ind, la, pos)), flip_0)
#     cv2.imwrite(os.path.join(dest_wrong_root, "{}_{}_{}.jpg".format(50000 + ind, la, 20 - pos)), flip_1)
#     cv2.imwrite(os.path.join(dest_wrong_root, "{}_{}_{}.jpg".format(60000 + ind, la, 20 - pos)), flip_2)


# # 转换成label
# def de_label(diff_inds=None):
#     label = "11111111111111111"
#     if not diff_inds is None:
#         for ind in diff_inds:
#             label = label[:ind] + "0" + label[ind + 1:]
#     return label
#
# ori_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/unlabeled/after_extend"
# ori_train_root = os.path.join(ori_root, "train")
# ori_train_right_root = os.path.join(ori_train_root, "right")
# ori_train_wrong_root = os.path.join(ori_train_root, "wrong")
# ori_val_root = os.path.join(ori_root, "val")
# ori_val_right_root = os.path.join(ori_val_root, "right")
# ori_val_wrong_root = os.path.join(ori_val_root, "wrong")
# dest_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/labeled"
# dest_train_root = os.path.join(dest_root, "train_and_val")
# dest_val_root = os.path.join(dest_root, "test")
#
# fname_list = os.listdir(ori_train_right_root)
# ori_path_list = []
# dest_path_list = []
# for fn in fname_list:
#     ori_path_list.append(os.path.join(ori_train_right_root, fn))
#     ind = fn.split("_")[0]
#     label = de_label()
#     dest_path_list.append(os.path.join(dest_train_root, "{}_{}.jpg".format(ind, label)))
# copy_files(ori_path_list, dest_path_list)
#
# fname_list = os.listdir(ori_train_wrong_root)
# ori_path_list = []
# dest_path_list = []
# for fn in fname_list:
#     ori_path_list.append(os.path.join(ori_train_wrong_root, fn))
#     ind, pos = fn.split("_")[0], int(fn.split("_")[2].split(".")[0])
#     label = de_label([pos-2])
#     dest_path_list.append(os.path.join(dest_train_root, "{}_{}.jpg".format(ind, label)))
# copy_files(ori_path_list, dest_path_list)
#
# fname_list = os.listdir(ori_val_right_root)
# ori_path_list = []
# dest_path_list = []
# for fn in fname_list:
#     ori_path_list.append(os.path.join(ori_val_right_root, fn))
#     ind = fn.split("_")[0]
#     label = de_label()
#     dest_path_list.append(os.path.join(dest_val_root, "{}_{}.jpg".format(ind, label)))
# copy_files(ori_path_list, dest_path_list)
#
# fname_list = os.listdir(ori_val_wrong_root)
# ori_path_list = []
# dest_path_list = []
# for fn in fname_list:
#     ori_path_list.append(os.path.join(ori_val_wrong_root, fn))
#     ind, pos = fn.split("_")[0], int(fn.split("_")[2].split(".")[0])
#     label = de_label([pos-2])
#     dest_path_list.append(os.path.join(dest_val_root, "{}_{}.jpg".format(ind, label)))
# copy_files(ori_path_list, dest_path_list)


# # 生成infer数据集并转换为label
# def de_label(diff_inds=None):
#     label = "11111111111111111"
#     if not diff_inds is None:
#         for ind in diff_inds:
#             label = label[:ind] + "0" + label[ind + 1:]
#     return label
#
# ori_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/CNN_LSTM_CTC_Tensorflow/unlabeled/before_extend/all"
# ori_right_root = os.path.join(ori_root, "right")
# ori_wrong_root = os.path.join(ori_root, "wrong")
# dest_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/CNN_LSTM_CTC_Tensorflow/labeled/infer"
#
# ori_path_list = []
# dest_path_list = []
#
# fname_list = os.listdir(ori_right_root)
# for fn in fname_list:
#     ori_path_list.append(os.path.join(ori_right_root, fn))
#     ind = fn.split("_")[0]
#     label = de_label()
#     dest_path_list.append(os.path.join(dest_root, "{}_{}.jpg".format(ind, label)))
#
# fname_list = os.listdir(ori_wrong_root)
# for fn in fname_list:
#     ori_path_list.append(os.path.join(ori_wrong_root, fn))
#     ind, pos = fn.split("_")[0], int(fn.split("_")[2].split(".")[0])
#     label = de_label([pos-2])
#     dest_path_list.append(os.path.join(dest_root, "{}_{}.jpg".format(ind, label)))
#
# copy_files(ori_path_list, dest_path_list)


# 生成交叉验证数据集
ori_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/labeled/train_and_val"
dest_root = "/home/csbhr/workspace/python/python_data/Seal_Detection/labeled/cross_validation"
dest_group1_root = os.path.join(dest_root, "group_1")
dest_group2_root = os.path.join(dest_root, "group_2")
dest_group3_root = os.path.join(dest_root, "group_3")
dest_group4_root = os.path.join(dest_root, "group_4")
dest_group5_root = os.path.join(dest_root, "group_5")

fname_list = os.listdir(ori_root)
random.shuffle(fname_list)

unit_len = int(len(fname_list) / 5)
fname_list_group1 = fname_list[:unit_len]
fname_list_group2 = fname_list[unit_len:unit_len * 2]
fname_list_group3 = fname_list[unit_len * 2:unit_len * 3]
fname_list_group4 = fname_list[unit_len * 3:unit_len * 4]
fname_list_group5 = fname_list[unit_len * 4:]

train_list = fname_list_group1 + fname_list_group2 + fname_list_group3 + fname_list_group4
val_list = fname_list_group5
group1_dict = {
    "ori_train": [os.path.join(ori_root, fn) for fn in train_list],
    "dest_train": [os.path.join(dest_group1_root, "train", fn) for fn in train_list],
    "ori_val": [os.path.join(ori_root, fn) for fn in val_list],
    "dest_val": [os.path.join(dest_group1_root, "val", fn) for fn in val_list]
}
train_list = fname_list_group1 + fname_list_group2 + fname_list_group3 + fname_list_group5
val_list = fname_list_group4
group2_dict = {
    "ori_train": [os.path.join(ori_root, fn) for fn in train_list],
    "dest_train": [os.path.join(dest_group2_root, "train", fn) for fn in train_list],
    "ori_val": [os.path.join(ori_root, fn) for fn in val_list],
    "dest_val": [os.path.join(dest_group2_root, "val", fn) for fn in val_list]
}
train_list = fname_list_group1 + fname_list_group2 + fname_list_group4 + fname_list_group5
val_list = fname_list_group3
group3_dict = {
    "ori_train": [os.path.join(ori_root, fn) for fn in train_list],
    "dest_train": [os.path.join(dest_group3_root, "train", fn) for fn in train_list],
    "ori_val": [os.path.join(ori_root, fn) for fn in val_list],
    "dest_val": [os.path.join(dest_group3_root, "val", fn) for fn in val_list]
}
train_list = fname_list_group1 + fname_list_group3 + fname_list_group4 + fname_list_group5
val_list = fname_list_group2
group4_dict = {
    "ori_train": [os.path.join(ori_root, fn) for fn in train_list],
    "dest_train": [os.path.join(dest_group4_root, "train", fn) for fn in train_list],
    "ori_val": [os.path.join(ori_root, fn) for fn in val_list],
    "dest_val": [os.path.join(dest_group4_root, "val", fn) for fn in val_list]
}
train_list = fname_list_group2 + fname_list_group3 + fname_list_group4 + fname_list_group5
val_list = fname_list_group1
group5_dict = {
    "ori_train": [os.path.join(ori_root, fn) for fn in train_list],
    "dest_train": [os.path.join(dest_group5_root, "train", fn) for fn in train_list],
    "ori_val": [os.path.join(ori_root, fn) for fn in val_list],
    "dest_val": [os.path.join(dest_group5_root, "val", fn) for fn in val_list]
}

copy_files(group1_dict["ori_train"], group1_dict["dest_train"])
copy_files(group1_dict["ori_val"], group1_dict["dest_val"])
copy_files(group2_dict["ori_train"], group2_dict["dest_train"])
copy_files(group2_dict["ori_val"], group2_dict["dest_val"])
copy_files(group3_dict["ori_train"], group3_dict["dest_train"])
copy_files(group3_dict["ori_val"], group3_dict["dest_val"])
copy_files(group4_dict["ori_train"], group4_dict["dest_train"])
copy_files(group4_dict["ori_val"], group4_dict["dest_val"])
copy_files(group5_dict["ori_train"], group5_dict["dest_train"])
copy_files(group5_dict["ori_val"], group5_dict["dest_val"])
