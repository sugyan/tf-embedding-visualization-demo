import os
import tensorflow as tf
from tensorflow.models.image.imagenet import classify_image

classify_image.FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '/tmp/imagenet',
                           """Path to model.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 1,
                            """Display this many predictions.""")


def main(argv=None):
    classify_image.maybe_download_and_extract()
    classify_image.create_graph()
    with tf.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        for filename in os.listdir('images'):
            if not filename.endswith('.JPG'):
                continue
            with open(os.path.join('images', filename), 'rb') as f:
                output = sess.run(pool3, {'DecodeJpeg/contents:0': f.read()})
            print(output.shape)


if __name__ == '__main__':
    tf.app.run()
