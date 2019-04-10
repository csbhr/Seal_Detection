import numpy as np
import os
import utils
import pickle


def pickle_dump(file_path, content_list):
    '''
    dump the list into the file using pickle module
    :param file_path: the file's path
    :param content_list: the list of content
    :return: none
    '''
    with open(file_path, "wb") as f:
        pickle.dump(content_list, f)


def pickle_load(file_path):
    '''
    load the file's content into list using pickle module
    :param file_path: the file's path
    :return: the list of content
    '''
    with open(file_path, "rb") as f:
        content_list = pickle.load(f)
    return content_list


FLAGS = utils.FLAGS

result_dict = pickle_load(os.path.join(FLAGS.infer_result_dir, "infer_result_test.data"))
# result_dict = pickle_load(os.path.join(FLAGS.infer_result_dir, "infer_result_test_all.data"))
print("# 计算测试集准确度")
# print("# 计算所有数据集准确度")

labels = result_dict["labels"]
probs = result_dict["probs"]
preds = result_dict["preds"]

sample_num = len(labels)






# 总体准确度
right_num = 0
for i in range(sample_num):
    sum_la = np.sum(labels[i])
    sum_pred = np.sum(preds[i])
    re_la = sum_la == 17
    re_pred = sum_pred == 17
    if re_la == re_pred:
        right_num += 1
total_accuracy = float(right_num)/float(sample_num)
print()
print("total_accuracy: ", total_accuracy)
print("numbers（正确数/总数）: {}/{}".format(right_num, sample_num))
print("说明：只要判定出是否有错误字符，并不判定错误字符位置！")

# 位置准确度
right_num = 0
for i in range(sample_num):
    sub = np.abs(np.subtract(labels[i], preds[i]))
    sum = np.sum(sub)
    if sum == 0:
        right_num += 1
position_accuracy = float(right_num)/float(sample_num)

print()
print("position_accuracy: ", position_accuracy)
print("numbers（正确数/总数）: {}/{}".format(right_num, sample_num))
print("说明：判定出是否有错误字符，并且判定错误字符位置，只有预测出的错误字符位置全部正确才算预测正确！")
