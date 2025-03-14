import tensorflow as tf

writer = tf.summary.create_file_writer('logs')
with writer.as_default():
    tf.summary.scalar('test_metric', 0.1, step=1)
    tf.summary.scalar('test_metric', 0.2, step=2)
    tf.summary.scalar('test_metric', 0.3, step=3)

