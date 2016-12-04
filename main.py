import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.models.image.imagenet import classify_image
from tensorflow.contrib.tensorboard.plugins import projector

classify_image.FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '/tmp/imagenet',
                           """Path to model.""")


def main(argv=None):
    classify_image.maybe_download_and_extract()
    classify_image.create_graph()
    basedir = os.path.dirname(__file__)
    with tf.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        jpeg_data = tf.placeholder(tf.string)
        thumbnail = tf.cast(tf.image.resize_images(tf.image.decode_jpeg(jpeg_data), [100, 100]), tf.uint8)
        outputs = []
        files = []
        images = []
        for filename in os.listdir('images'):
            if not filename.endswith('.JPG'):
                continue
            print('process %s...' % filename)
            files.append(filename)
            with open(os.path.join(basedir, 'images', filename), 'rb') as f:
                data = f.read()
                results = sess.run([pool3, thumbnail], {'DecodeJpeg/contents:0': data, jpeg_data: data})
                outputs.append(results[0])
                images.append(results[1])

        embedding_var = tf.Variable(tf.pack([tf.squeeze(x) for x in outputs], axis=0), trainable=False, name='pool3')
        # prepare projector config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        summary_writer = tf.train.SummaryWriter(os.path.join(basedir, 'logdir'))
        # link metadata
        metadata_path = os.path.join(basedir, 'logdir', 'metadata.tsv')
        with open(metadata_path, 'w') as f:
            for name in files:
                f.write('%s\n' % name)
        embedding.metadata_path = metadata_path
        # write to sprite image file
        image_path = os.path.join(basedir, 'logdir', 'sprite.jpg')
        size = int(math.sqrt(len(images))) + 1
        while len(images) < size * size:
            images.append(np.zeros((100, 100, 3), dtype=np.uint8))
        rows = []
        for i in range(size):
            rows.append(tf.concat(1, images[i*size:(i+1)*size]))
        jpeg = tf.image.encode_jpeg(tf.concat(0, rows))
        with open(image_path, 'wb') as f:
            f.write(sess.run(jpeg))
        embedding.sprite.image_path = image_path
        embedding.sprite.single_image_dim.extend([100, 100])
        # save embedding_var
        projector.visualize_embeddings(summary_writer, config)
        sess.run(tf.variables_initializer([embedding_var]))
        saver = tf.train.Saver([embedding_var])
        saver.save(sess, os.path.join(basedir, 'logdir', 'model.ckpt'))


if __name__ == '__main__':
    tf.app.run()
