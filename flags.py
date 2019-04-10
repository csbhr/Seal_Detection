import tensorflow as tf

tf.flags.DEFINE_string('mode', 'infer', 'train or infer')
tf.flags.DEFINE_integer('num_gpus', 1, 'num of gpus')

tf.flags.DEFINE_integer('image_height', 80, 'image height')
tf.flags.DEFINE_integer('image_width', 200, 'image width')
tf.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.flags.DEFINE_integer('out_channels', 128, 'output channels of last layer in CNN')
tf.flags.DEFINE_integer('cnn_count', 3, 'count of cnn module to extract image features.')
tf.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.flags.DEFINE_float('output_keep_prob', 0.5, 'output_keep_prob in lstm')
tf.flags.DEFINE_integer('num_epochs', 500, 'maximum epochs')
tf.flags.DEFINE_integer('batch_size', 1, 'the batch_size')
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
                       '/home/csbhr/workspace/python/python_data/Seal_Detection/labeled/cross_validation/group_1/train',
                       'the train data dir')
tf.flags.DEFINE_string('val_dir',
                       '/home/csbhr/workspace/python/python_data/Seal_Detection/labeled/cross_validation/group_1/val',
                       'the val data dir')
tf.flags.DEFINE_string('infer_dir',
                       '/home/csbhr/workspace/python/python_data/Seal_Detection/labeled/test',
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