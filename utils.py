import os
import numpy as np
import cv2
import flags

FLAGS = flags.FLAGS

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
