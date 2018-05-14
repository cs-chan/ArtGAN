import tensorflow as tf
import os


records_default = [[''], [0]]
IMAGENET_MEAN = [103.939, 116.779, 123.68]


def read_csv(filename, directory, num_epoch=None, records=None):
    if records is None:
        records = records_default
    # Read examples and labels from file
    filename_queue = tf.train.string_input_producer([os.path.join(directory, filename)],
                                                    num_epochs=num_epoch, shuffle=True)
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, record_defaults=records)
    im_path = tf.stack(decoded[0])
    if len(decoded) > 1:
        label = tf.stack(decoded[1:])
    else:
        label = [0]
    return im_path, label


def decode_jpg(filename, directory, center=False, crop=None, flip=False, resize=None, ratio=False, filename_offset='',
               normalize='imagenet'):
    # Read image
    im_content = tf.read_file(directory + filename_offset + filename)
    example = tf.image.decode_jpeg(im_content, channels=3)
    # preprocessing
    example = tf.cast(example[:, :, ::-1], tf.float32)
    if normalize is 'imagenet':
        example = example - IMAGENET_MEAN
    elif normalize:
        example = example / 127.5 - 1.
    # cropping
    if crop:
        shape = tf.shape(example)
        if ratio:
            assert isinstance(crop, list)
            crop_h = crop[0]
            crop_w = crop[1]
        else:
            assert isinstance(crop, int)
            shortest = tf.cond(tf.less(shape[0], shape[1]), lambda: shape[0], lambda: shape[1])
            crop_h = tf.cond(tf.less_equal(shortest, tf.constant(crop)), lambda: shortest, lambda: tf.constant(crop))
            crop_w = crop_h
        if center:
            example = tf.image.resize_image_with_crop_or_pad(example, crop_h, crop_w)
        else:
            example = tf.random_crop(example, [crop_h, crop_w, 3])
    # resize
    if resize:
        assert isinstance(resize, (int, float))
        new_size = tf.stack([resize, resize])
        example = tf.image.resize_images(example, new_size)
    # random horizontal flip
    if flip:
        example = tf.image.random_flip_left_right(example)
    return tf.transpose(example, [2, 0, 1])


def input_pipeline(filenames, directory, batchsize, num_epoch=None, records=None, center=False, crop=None,
                   flip=False, resize=None, ratio=False, filename_offset='', normalize='imagenet'):
    if records is None:
        records = records_default
    imname, label = read_csv(filenames, directory, num_epoch, records)
    example = decode_jpg(imname, directory, center, crop, flip, resize, ratio, filename_offset, normalize)
    min_after_dequeue = batchsize * 3
    capacity = 10 * batchsize
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batchsize, capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=4,
        shapes=((3, resize, resize), (1,)), allow_smaller_final_batch=True
    )
    return example_batch, label_batch
