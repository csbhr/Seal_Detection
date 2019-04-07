import os
import cv2 as cv


# 获取路径path中所有文件和文件夹名
def get_allFile_byPath(path):
    return os.listdir(path)


def threshold_img_1(ori_path, fname, indic):
    '''
    二值化图像
    思路：
        首先把r通道低于180的像素的三通道设为255（即设为白色）
        再讲b通道、g通道均大于180的像素的三通道设为0（即把bg偏白的像素设为黑色）
    :param ori_path: 原图像路径
    :param fname: 原图像名
    :param indic: 指示表，查表用，用于提高速度
    :return: 二值化图像
    '''
    img = cv.imread(ori_path + "/" + fname, cv.IMREAD_COLOR)
    height, width = img[:, :, 0].shape
    for i in range(height):
        for j in range(width):
            if not indic[img[i, j, 2]]:
                img[i, j, 0] = 255
                img[i, j, 1] = 255
                img[i, j, 2] = 255
    for i in range(height):
        for j in range(width):
            if indic[img[i, j, 0]] and indic[img[i, j, 1]]:
                img[i, j, 0] = 0
                img[i, j, 1] = 0
                img[i, j, 2] = 0
            else:
                img[i, j, 2] = 255
    return img[:, :, 2]


def threshold_cut_img(ori_path, fname, indic):
    '''
    二值化已经裁剪好的图像
    思路：
        首先把r通道低于180的像素的三通道设为255（即设为白色）
        再讲b通道、g通道均大于180的像素的三通道设为0（即把bg偏白的像素设为黑色）
    :param ori_path: 原图像路径
    :param fname: 原图像名
    :param indic: 指示表，查表用，用于提高速度
    :return: 二值化的已经裁剪好的图像
    '''
    img = cv.imread(ori_path + "/" + fname, cv.IMREAD_COLOR)
    height, width = img[:, :, 0].shape
    for i in range(height):
        for j in range(width):
            if indic[img[i, j, 2]]:
                img[i, j, 0] = 0
                img[i, j, 1] = 0
                img[i, j, 2] = 0
            else:
                img[i, j, 0] = 255
                img[i, j, 1] = 255
                img[i, j, 2] = 255
    return img


def threshold_img(ori_img):
    '''
    :param ori_img: the origin image
    :return: the threshold image
    '''
    img_ori = ori_img.copy()
    indic_half = 180
    indic = [False for i in range(indic_half)]
    indic.extend([True for i in range(256 - indic_half)])
    height, width = img_ori[:, :, 0].shape
    for i in range(height):
        for j in range(width):
            if not indic[img_ori[i, j, 2]]:
                img_ori[i, j, 0] = 255
                img_ori[i, j, 1] = 255
                img_ori[i, j, 2] = 255
    for i in range(height):
        for j in range(width):
            if indic[img_ori[i, j, 0]] and indic[img_ori[i, j, 1]]:
                img_ori[i, j, 0] = 0
                img_ori[i, j, 1] = 0
                img_ori[i, j, 2] = 0
            else:
                img_ori[i, j, 2] = 255
    return img_ori[:, :, 2]


def action():
    # ori_path = "D:/WorkSpace/python/project_data/Seal_Detection/vin"
    # store_path = "D:/WorkSpace/python/project_data/Seal_Detection/thresholdIMG"
    ori_path = "D:/WorkSpace/python/project_data/Seal_Detection/cutIMG"
    store_path = "D:/WorkSpace/python/project_data/Seal_Detection/thresholdCutIMG"
    indic_half = 180
    indic = [False for i in range(indic_half)]
    indic.extend([True for i in range(256 - indic_half)])
    filelist = get_allFile_byPath(ori_path)
    # for fname in filelist[3141:6400]:
    # for fname in filelist[9498:12800]:
    # for fname in filelist[15911:19200]:
    # for fname in filelist[22033:25600]:
    # for fname in filelist[28693:32000]:
    for fname in filelist:
        img_r = threshold_img_1(ori_path, fname, indic)
        cv.imwrite(store_path + "/" + fname, img_r)
        print("--store image " + fname + "--------")
