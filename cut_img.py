import os
import cv2 as cv
import numpy as np
import file_tools as ftools


# 获取路径path中所有文件和文件夹名
def get_allFile_byPath(path):
    return os.listdir(path)


def mean_filter(sou, filter_size=3):
    '''
    均值滤波，对一维数组进行均值滤波
    :param sou: 源数组
    :param filter_size: 窗口大小
    :return: 均值滤波后的数组
    '''
    result = sou.copy()
    margin = int((filter_size - 1) / 2)
    for i in range(sou.shape[0])[margin:-margin]:
        sum = 0
        for j in range(filter_size):
            sum = sum + sou[i - margin + j]
        result[i] = sum / filter_size
    return result


def cut_img_1(img_ori, img_thre):
    '''
    水平切割：
        水平投影
        使用均值滤波，并使用均值（*2）进行筛选
        自下而上的选择第一个红色区块
    垂直切割：
        垂直投影
        使用均值滤波，并使用均值进行筛选
        从两端检测，去除两端空白
    :param ori_path: 原图像路径
    :param thre_path: 二值化图像路径
    :param fname: 图像名称
    :return: 在二值化图像、原图上切割的图片
    '''
    # img_thre = cv.imread(thre_path + "/" + fname, cv.IMREAD_GRAYSCALE)  # 二值化图像
    # img_ori = cv.imread(ori_path + "/" + fname, cv.IMREAD_COLOR)  # 原图

    # 计算水平投影
    projection_x = np.sum(img_thre, axis=1)
    # 均值滤波
    projection_x = mean_filter(projection_x, filter_size=5)
    # 计算均值（*2）
    pro_mean_x = np.mean(projection_x) * 2
    # 根据均值（*2）水平筛选
    for i in range(projection_x.shape[0]):
        if projection_x[i] < pro_mean_x:
            img_thre[i] = 0
            projection_x[i] = 0
    # 确定水平切割位置
    positions_x = [0, 0]
    flag = 1
    for i in range(projection_x.shape[0])[::-1]:
        if flag < 0:
            break
        if not projection_x[i] == 0:
            if flag == 1:
                positions_x[flag] = i
                flag = flag - 1
        if projection_x[i] == 0:
            if flag == 0:
                positions_x[flag] = i + 1
                flag = flag - 1
    # 保证水平切割时的高度大于60
    if positions_x[1] - positions_x[0] < 60:
        positions_x[0] = positions_x[1] - 100
    # 对二值化图像水平切割，以便于垂直投影
    img_cut_x = img_thre[positions_x[0]:positions_x[1]].copy()

    # 计算垂直投影、均值
    projection_y = np.sum(img_cut_x, axis=0)
    projection_y = mean_filter(projection_y)
    pro_mean_y = np.mean(projection_y)
    # 根据均值（*1.5）垂直筛选
    for i in range(projection_y.shape[0]):
        if projection_y[i] < pro_mean_y:
            img_cut_x[:, i] = 0
            projection_y[i] = 0
    # 确定垂直切割位置
    positions_y = [0, 0]
    for i in range(projection_y.shape[0]):
        if not projection_y[i] == 0:
            positions_y[0] = i
            break
    for i in range(projection_y.shape[0])[::-1]:
        if not projection_y[i] == 0:
            positions_y[1] = i
            break

    # 二值化图像切割
    img_thre_cut_final = img_thre[positions_x[0]:positions_x[1], positions_y[0]:positions_y[1]].copy()
    # 原图切割
    img_ori_cut_final = img_ori[positions_x[0]:positions_x[1], positions_y[0]:positions_y[1]].copy()
    return img_thre_cut_final, img_ori_cut_final


def cut_img_2(img_ori, img_thre):
    '''
    把所有的区块全部选出来：
        使用的还是（cut_img_1）的方法
    对图片进行筛选：
        去除比例小于3的
        去除高度小于50的
        从下往上选取第一个
    :param ori_path: 原图像路径
    :param thre_path: 二值化图像路径
    :param fname: 图像名称
    :return: 在二值化图像、原图上切割的图片
    '''
    # img_thre = cv.imread(thre_path + "/" + fname, cv.IMREAD_GRAYSCALE)
    # img_ori = cv.imread(ori_path + "/" + fname, cv.IMREAD_COLOR)

    # 计算水平投影
    projection_x = np.sum(img_thre, axis=1)
    # 均值滤波
    projection_x = mean_filter(projection_x)
    # 计算均值（*2）
    pro_mean_x = np.mean(projection_x) * 2
    # 根据均值（*2）筛选
    for i in range(projection_x.shape[0]):
        if projection_x[i] < pro_mean_x:
            img_thre[i] = 0
            projection_x[i] = 0
    # 存储水平切割的多个图片
    cut_x_img_thre_list = []
    cut_x_img_ori_list = []
    # 水平切割二值化图片、原图，并分别放在cut_x_img_thre_list、cut_x_img_ori_list中
    pos_x = [0, 0]
    height = projection_x.shape[0]
    i = 0
    while i < height:
        if not projection_x[i] == 0:
            pos_x[0] = i
            pos_x[1] = height
            j = i
            while j < height:
                if projection_x[j] == 0:
                    pos_x[1] = j
                    i = j
                    break
                j = j + 1
            img_thre_cut_temp = img_thre[pos_x[0]:pos_x[1]].copy()
            img_ori_cut_temp = img_ori[pos_x[0]:pos_x[1]].copy()
            cut_x_img_thre_list.append(img_thre_cut_temp)
            cut_x_img_ori_list.append(img_ori_cut_temp)
        i = i + 1

    # 存储垂直切割的多个图片
    cut_y_img_thre_list = []
    cut_y_img_ori_list = []
    # 垂直切割二值化图片、原图，并分别放在cut_y_img_thre_list、cut_y_img_ori_list中
    for cut_x_img_thre, cut_x_img_ori in zip(cut_x_img_thre_list, cut_x_img_ori_list):
        # 计算垂直投影
        projection_y = np.sum(cut_x_img_thre, axis=0)
        # 均值滤波
        projection_y = mean_filter(projection_y)
        # 计算均值
        pro_mean_y = np.mean(projection_y)
        # 根据均值筛选
        for i in range(projection_y.shape[0]):
            if projection_y[i] < pro_mean_y:
                cut_x_img_thre[:, i] = 0
                projection_y[i] = 0
        # 确定垂直切割位置
        pos_y = [0, 0]
        for i in range(projection_y.shape[0]):
            if not projection_y[i] == 0:
                pos_y[0] = i
                break
        for i in range(projection_y.shape[0])[::-1]:
            if not projection_y[i] == 0:
                pos_y[1] = i
                break
        img_thre_cut_temp = cut_x_img_thre[:, pos_y[0]:pos_y[1]].copy()
        img_ori_cut_temp = cut_x_img_ori[:, pos_y[0]:pos_y[1]].copy()
        cut_y_img_thre_list.append(img_thre_cut_temp)
        cut_y_img_ori_list.append(img_ori_cut_temp)

    # 筛选图片
    best_index = len(cut_y_img_thre_list) - 1
    if best_index < 0:
        return None, None  # 没有截取到图章
    for i in range(len(cut_y_img_thre_list))[::-1]:
        temp_rate = cut_y_img_thre_list[i].shape[1] / cut_y_img_thre_list[i].shape[0]
        if temp_rate < 3:  # 去除比例小于3的（去除较方的）
            continue
        if cut_y_img_thre_list[i].shape[0] < 50:  # 去除高度小于50的
            continue
        best_index = i  # 从下往上选取的第一个
        break
    return cut_y_img_thre_list[best_index], cut_y_img_ori_list[best_index]


def cut_img_3(img_ori, img_thre):
    '''
    把所有的区块全部选出来：
        使用的还是（cut_img_1）的方法
    对图片进行筛选：
        去除高度小于50的
        根据统计出的长宽比的权重进行筛选，选出权重最大的
    :param ori_path: 原图像路径
    :param thre_path: 二值化图像路径
    :param fname: 图像名称
    :return: 在二值化图像、原图上切割的图片
    '''
    # img_thre = cv.imread(thre_path + "/" + fname, cv.IMREAD_GRAYSCALE)
    # img_ori = cv.imread(ori_path + "/" + fname, cv.IMREAD_COLOR)

    # 计算水平投影
    projection_x = np.sum(img_thre, axis=1)
    # 均值滤波
    projection_x = mean_filter(projection_x)
    # 计算均值（*2）
    pro_mean_x = np.mean(projection_x) * 2
    # 根据均值（*2）筛选
    for i in range(projection_x.shape[0]):
        if projection_x[i] < pro_mean_x:
            img_thre[i] = 0
            projection_x[i] = 0
    # 存储水平切割的多个图片
    cut_x_img_thre_list = []
    cut_x_img_ori_list = []
    # 水平切割二值化图片、原图，并分别放在cut_x_img_thre_list、cut_x_img_ori_list中
    pos_x = [0, 0]
    height = projection_x.shape[0]
    i = 0
    while i < height:
        if not projection_x[i] == 0:
            pos_x[0] = i
            pos_x[1] = height
            j = i
            while j < height:
                if projection_x[j] == 0:
                    pos_x[1] = j
                    i = j
                    break
                j = j + 1
            img_thre_cut_temp = img_thre[pos_x[0]:pos_x[1]].copy()
            img_ori_cut_temp = img_ori[pos_x[0]:pos_x[1]].copy()
            cut_x_img_thre_list.append(img_thre_cut_temp)
            cut_x_img_ori_list.append(img_ori_cut_temp)
        i = i + 1

    # 存储垂直切割的多个图片
    cut_y_img_thre_list = []
    cut_y_img_ori_list = []
    # 垂直切割二值化图片、原图，并分别放在cut_y_img_thre_list、cut_y_img_ori_list中
    for cut_x_img_thre, cut_x_img_ori in zip(cut_x_img_thre_list, cut_x_img_ori_list):
        # 计算垂直投影
        projection_y = np.sum(cut_x_img_thre, axis=0)
        # 均值滤波
        projection_y = mean_filter(projection_y)
        # 计算均值
        pro_mean_y = np.mean(projection_y)
        # 根据均值筛选
        for i in range(projection_y.shape[0]):
            if projection_y[i] < pro_mean_y:
                cut_x_img_thre[:, i] = 0
                projection_y[i] = 0
        # 确定垂直切割位置
        pos_y = [0, 0]
        for i in range(projection_y.shape[0]):
            if not projection_y[i] == 0:
                pos_y[0] = i
                break
        for i in range(projection_y.shape[0])[::-1]:
            if not projection_y[i] == 0:
                pos_y[1] = i
                break
        img_thre_cut_temp = cut_x_img_thre[:, pos_y[0]:pos_y[1]].copy()
        img_ori_cut_temp = cut_x_img_ori[:, pos_y[0]:pos_y[1]].copy()
        cut_y_img_thre_list.append(img_thre_cut_temp)
        cut_y_img_ori_list.append(img_ori_cut_temp)

    # 筛选图片
    remove_list = []
    for i in range(len(cut_y_img_thre_list)):
        if cut_y_img_thre_list[i].shape[0] < 50:  # 去除高度小于50的
            remove_list.append(i)

    info = ftools.pickle_load("weight.txt")  # 读取权重信息
    weights = info[0]
    otherwise = info[1]
    img_weight_info = []
    for i in range(len(cut_y_img_thre_list)):  # 计算各切片权重
        if i not in remove_list:
            rate = float(cut_y_img_thre_list[i].shape[1]) / cut_y_img_thre_list[i].shape[0]
            rate = int(rate / 0.1)
            if rate >= 200:
                img_weight_info.append([i, otherwise])
            else:
                img_weight_info.append([i, weights[rate]])
    if len(img_weight_info) <= 0:  # 没有截取到图章
        return None, None
    best_index = 0
    for i in range(len(img_weight_info)):
        if img_weight_info[i][1] > img_weight_info[best_index][1]:
            best_index = i
    best_index = img_weight_info[best_index][0]
    return cut_y_img_thre_list[best_index], cut_y_img_ori_list[best_index]


def cut_img(img_ori, img_thre):
    '''
    :param img_ori: the origin image
    :param img_thre: the threshold image
    :return: the cut image
    '''

    cut_thre_img, cut_ori_img = cut_img_2(img_ori, img_thre)

    return cut_ori_img


def action():
    ori_path = "D:/WorkSpace/python/project_data/Seal_Detection/vin"
    thre_path = "D:/WorkSpace/python/project_data/Seal_Detection/thresholdIMG"
    store_path = "D:/WorkSpace/python/project_data/Seal_Detection/cutIMG_algorithm3"
    filelist = get_allFile_byPath(ori_path)
    # for fname in filelist[3141:6400]:
    # for fname in filelist[9498:12800]:
    # for fname in filelist[15911:19200]:
    # for fname in filelist[22033:25600]:
    # for fname in filelist[28693:32000]:
    for fname in filelist:
        img_thre = cv.imread(thre_path + "/" + fname, cv.IMREAD_GRAYSCALE)
        img_ori = cv.imread(ori_path + "/" + fname, cv.IMREAD_COLOR)
        # img_thre_cut_final, img_ori_cut_final = cut_img_1(img_ori, img_thre)  # 切割算法
        # img_thre_cut_final, img_ori_cut_final = cut_img_2(img_ori, img_thre)  # 切割算法
        img_thre_cut_final, img_ori_cut_final = cut_img_3(img_ori, img_thre)  # 切割算法
        cv.imwrite(store_path + "/" + fname, img_ori_cut_final)
        print("--store image " + fname + "--------")
