import os
import numpy as np
import tensorflow as tf
import cv2


tf.flags.DEFINE_string('mode', 'train', 'train or infer')
tf.flags.DEFINE_integer('num_gpus', 1, 'num of gpus')

tf.flags.DEFINE_integer('image_height', 80, 'image height')
tf.flags.DEFINE_integer('image_width', 200, 'image width')
tf.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.flags.DEFINE_integer('out_channels', 128, 'output channels of last layer in CNN')
tf.flags.DEFINE_integer('cnn_count', 3, 'count of cnn module to extract image features.')
tf.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.flags.DEFINE_float('output_keep_prob', 0.5, 'output_keep_prob in lstm')
tf.flags.DEFINE_integer('num_epochs', 300, 'maximum epochs')
tf.flags.DEFINE_integer('batch_size', 8, 'the batch_size')
tf.flags.DEFINE_integer('save_steps', 1000, 'the step to save checkpoint')
tf.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.flags.DEFINE_integer('validation_steps', 50, 'the step to validation')
tf.flags.DEFINE_integer('train_with_val_steps', 10, 'train model with val dataset every train_with_val_steps epoch')
tf.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')

tf.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')
tf.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
tf.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')
tf.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.flags.DEFINE_string('train_dir',
                       '/home/csbhr/workspace/python/python_data/Seal_Detection/labeled/cross_validation/group_3/train',
                       'the train data dir')
tf.flags.DEFINE_string('val_dir',
                       '/home/csbhr/workspace/python/python_data/Seal_Detection/labeled/cross_validation/group_3/val',
                       'the val data dir')
tf.flags.DEFINE_string('infer_dir',
                       '/home/csbhr/workspace/python/python_data/Seal_Detection/labeled/test_all',
                       'the infer data dir')
tf.flags.DEFINE_string('log_dir',
                       '/home/csbhr/workspace/python/python_data/Seal_Detection/log',
                       'the logging dir')
tf.flags.DEFINE_string('checkpoint_dir',
                       '/home/csbhr/workspace/python/python_data/Seal_Detection/checkpoint',
                       'the checkpoint dir')
tf.flags.DEFINE_string('infer_result_dir',
                       '/home/csbhr/workspace/python/python_data/Seal_Detection',
                       'the infer result dir')

FLAGS = tf.flags.FLAGS

label_len = 17
num_class = 2


class DataIterator:
    def __init__(self, data_dir):
        self.image = []
        self.labels = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                im = cv2.imread(image_name, cv2.IMREAD_COLOR)[:, :, 1].astype(np.float32) / 255.

                # resize to same height, different width will consume time on padding
                im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height), interpolation=cv2.INTER_CUBIC)
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                self.image.append(im)

                # image is named as /.../<folder>/00000_abcd.png
                code = image_name.split('/')[-1].split('_')[1].split('.')[0]
                codes = list(code)
                label = np.zeros([label_len, num_class])
                for i in range(len(codes)):
                    label[i, int(codes[i])] = 1.

                self.labels.append(label)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    def input_index_generate_batch(self, index=None):
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        return image_batch, label_batch
