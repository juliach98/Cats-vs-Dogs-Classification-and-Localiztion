import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from os import listdir
from os.path import *
from utils import mIoU


def preprocess_image(img, image_size):
    image = tf.image.resize(img, [image_size, image_size])
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image


def postprocess_image(img, results):
    classes = {0: 'cat',
               1: 'dog'}

    class_probs, bounding_box = results

    class_index = np.argmax(class_probs)
    class_label = classes[int(class_index)]

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bounding_box[0]
    x1 = int(w * x1)
    x2 = int(w * x2)
    y1 = int(h * y1)
    y2 = int(h * y2)

    return class_label, (x1, y1, x2, y2), class_probs[0][class_index] * 100


def predict_and_draw(test_dir, filename, model, image_size, channels, scale):
    test_dir = abspath(test_dir)
    image = tf.io.read_file(join(test_dir, filename + '.jpg'))
    image = tf.image.decode_jpeg(image, channels)

    processed_image = preprocess_image(image, image_size)
    results = model.predict(processed_image)

    label, (x1, y1, x2, y2), confidence = postprocess_image(image, results)

    image = cv2.imread(join(test_dir, filename + '.jpg'))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 100), 2)
    cv2.putText(
        image,
        '{}, {:10.2f}%'.format(label, confidence),
        (x1, y2 + int(35 * scale)),
        cv2.FONT_HERSHEY_COMPLEX, scale,
        (0, 0, 255),
        2
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(image[:, :, ::-1])
    plt.show()


def predict_and_draw_all(test_dir, model, image_size, channels, scale=0.9):
    if isdir(test_dir):
        file_names = list(set([splitext(filename)[0] for filename in listdir(test_dir)]))
        for filename in file_names:
            predict_and_draw(test_dir, filename, model, image_size, channels, scale)
    else:
        dir = dirname(test_dir)
        filename = splitext(basename(test_dir))[0]
        predict_and_draw(dir, filename, model, image_size, channels, scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testdata_path', required=True, type=str,
                        help='Test data directory or path to image. '
                             'Example: D:\cats_dogs_dataset or D:\cats_dogs_dataset\dog.jpg.')
    parser.add_argument('--model_filename', required=True, type=str,
                        help='File name with model to load. Example: simpleCNN_0.h5.')
    parser.add_argument('--image_size', type=int, default=300)
    parser.add_argument('--channels', type=int, default=1)
    args = parser.parse_args()

    model = load_model(args.model_filename, compile=False)
    model.compile(loss={'label': 'sparse_categorical_crossentropy',
                        'box': 'mse'},
                  metrics={'label': 'accuracy',
                           'box': mIoU})
    predict_and_draw_all(args.testdata_path, model, args.image_size, args.channels)

