import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf
import pickle

import cnn_lstm_otc_ocr
import utils

# import helper

FLAGS = utils.FLAGS
label_len = utils.label_len
num_class = utils.num_class

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None, mode='train'):
    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()

    print('loading train data')
    train_feeder = utils.DataIterator(data_dir=train_dir)
    print('size: ', train_feeder.size)

    print('loading validation data')
    val_feeder = utils.DataIterator(data_dir=val_dir)
    print('size: {}\n'.format(val_feeder.size))

    num_train_samples = train_feeder.size  # 26000
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)  # example: 26000/100

    num_val_samples = val_feeder.size  # 8000
    num_batches_per_epoch_val = int(num_val_samples / FLAGS.batch_size)  # example: 8000/100
    shuffle_idx_val = np.random.permutation(num_val_samples)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess, ckpt)
                print('restore from checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx = np.random.permutation(num_train_samples)
            train_cost = 0
            start_time = time.time()
            batch_time = time.time()

            # the training part
            for cur_batch in range(num_batches_per_epoch):
                batch_time = time.time()
                indexs = [shuffle_idx[i % num_train_samples] for i in
                          range(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]
                # batch_inputs, _, batch_labels = train_feeder.input_index_generate_batch(indexs)
                batch_inputs, batch_labels = train_feeder.input_index_generate_batch(indexs)
                # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)
                feed = {model.inputs: batch_inputs,
                        model.labels: batch_labels}

                # if summary is needed
                summary_str, batch_cost, step, _ = \
                    sess.run([model.merged_summay, model.loss, model.global_step, model.train_op], feed_dict=feed)
                # calculate the cost
                train_cost += batch_cost * FLAGS.batch_size
                train_writer.add_summary(summary_str, step)
                if (cur_batch + 1) % 2 == 0:
                    print('batch', cur_batch, '/', num_batches_per_epoch, ' loss =', batch_cost, ' time',
                          time.time() - batch_time)
                # save the checkpoint
                if step % FLAGS.save_steps == 1:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    logger.info('save checkpoint at step {0}', format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)

                # train_err += the_err * FLAGS.batch_size
                # do validation
                if step % FLAGS.validation_steps == 0:
                    acc_batch_total = 0.
                    lastbatch_err = 0.
                    lr = 0
                    for j in range(num_batches_per_epoch_val):
                        indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
                                      range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
                        # val_inputs, _, val_labels = val_feeder.input_index_generate_batch(indexs_val)
                        val_inputs, val_labels = val_feeder.input_index_generate_batch(indexs_val)
                        val_feed = {model.inputs: val_inputs,
                                    model.labels: val_labels}

                        lastbatch_err, acc, lr = \
                            sess.run([model.loss, model.accuracy, model.lrn_rate], feed_dict=val_feed)

                        # print the decode result
                        # ori_labels = val_feeder.the_label(indexs_val)
                        # acc = utils.accuracy_calculation(ori_labels, dense_decoded,
                        #                                  ignore_value=-1, isPrint=True)
                        acc_batch_total += acc

                    accuracy = (acc_batch_total * FLAGS.batch_size) / num_val_samples

                    avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)

                    # train_err /= num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                          "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                          "lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                     cur_epoch + 1, FLAGS.num_epochs, accuracy, avg_train_cost,
                                     lastbatch_err, time.time() - start_time, lr))


def infer(img_path, mode='infer'):

    def load_img_path(img_path):
        fname_list = os.listdir(img_path)
        path_list = [os.path.join(img_path, fn) for fn in fname_list]
        return path_list

    imgList = load_img_path(img_path)

    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()

    total_steps = int(len(imgList) / FLAGS.batch_size)

    result_dict = {
        "labels": [],
        "probs": []
    }

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')

        for curr_step in range(total_steps):

            imgs_input = []
            for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
                result_dict["labels"].append(img.split("/")[-1].split("_")[1].split(".")[0])
                im = cv2.imread(img, cv2.IMREAD_COLOR)[:, :, 1].astype(np.float32) / 255.
                im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height), interpolation=cv2.INTER_CUBIC)
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                imgs_input.append(im)

            imgs_input = np.asarray(imgs_input)

            feed = {model.inputs: imgs_input}
            batch_prob = sess.run(model.prob, feed_dict=feed)
            result_dict["probs"].extend(batch_prob)

    labels = result_dict["labels"]
    probs = result_dict["probs"]
    result_dict = {
        "labels": [],
        "probs": [],
        "preds": []
    }
    for la_str in labels:
        la_array = np.zeros([label_len])
        la_list = list(la_str)
        for i in range(label_len):
            la_array[i] = float(la_list[i])
        result_dict["labels"].append(la_array)
    for pr in probs:
        pr_array = np.zeros([label_len])
        pred_array = np.zeros([label_len])
        for i in range(label_len):
            pr_array[i] = pr[i][0]/(pr[i][0]+pr[i][1])
            if pr_array[i] >= 0.5:
                pred_array[i] = 0.
            else:
                pred_array[i] = 1.
        result_dict["probs"].append(pr_array)
        result_dict["preds"].append(pred_array)

    pickle_dump(os.path.join(FLAGS.infer_result_dir, "infer_result.data"), result_dict)



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


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)

        elif FLAGS.mode == 'infer':
            infer(FLAGS.infer_dir, FLAGS.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
