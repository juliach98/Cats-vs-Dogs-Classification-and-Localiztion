import tensorflow as tf


def preprocess_data(data_dir, image_name, image_size, channels):
    image = tf.io.read_file(data_dir + '/' + image_name + '.jpg')
    image = tf.image.decode_jpeg(image, channels)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    txt = tf.io.read_file(data_dir + '/' + image_name + '.txt')
    txt = tf.strings.split([txt])[0]
    txt = tf.strings.to_number(txt, tf.int32)
    label = txt[0] - 1
    box = []
    box.append(txt[1] / width)
    box.append(txt[2] / height)
    box.append(txt[3] / width)
    box.append(txt[4] / height)

    image = tf.image.resize(image, [image_size, image_size])
    image /= 255.0

    return image, label, box


@tf.function
def preprocess(data_dir, image_name, image_size, channels):
    image, label, box = preprocess_data(data_dir, image_name, image_size, channels)
    return image, {'label': label, 'box': box}


def load_dataset(data_dir, filenames, batch_size, image_size, channels):
    loader = tf.data.Dataset.from_tensor_slices(filenames)

    preproc_data = lambda x: preprocess(data_dir, x, image_size, channels)
    AUTO = tf.data.experimental.AUTOTUNE
    loader = (
        loader
            .shuffle(len(filenames))
            .map(preproc_data, num_parallel_calls=AUTO)
            .batch(batch_size)
            .prefetch(AUTO)
        )

    return loader


def mIoU(box_true, box_pred):
    xA = tf.maximum(box_true[:, 0], box_pred[:, 0])
    yA = tf.maximum(box_true[:, 1], box_pred[:, 1])
    xB = tf.minimum(box_true[:, 2], box_pred[:, 2])
    yB = tf.minimum(box_true[:, 3], box_pred[:, 3])

    interArea = tf.maximum(0.0, xB - xA + 1) * tf.maximum(0.0, yB - yA + 1)

    box_true_area = (box_true[:, 2] - box_true[:, 0] + 1) * (box_true[:, 3] - box_true[:, 1] + 1)
    box_pred_area = (box_pred[:, 2] - box_pred[:, 0] + 1) * (box_pred[:, 3] - box_pred[:, 1] + 1)

    iou = interArea / (box_true_area + box_pred_area - interArea)

    return iou
