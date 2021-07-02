import argparse
from utils import *
from os.path import splitext
from os import listdir
from tensorflow.keras.models import load_model


def evaluate_model(valid_dir, model_filename, batch_size, image_size, channels):
    valid_filenames = list(set([splitext(filename)[0] for filename in listdir(valid_dir)]))

    testloader = load_dataset(valid_dir, valid_filenames, batch_size, image_size, channels)

    model = load_model(model_filename, compile=False)
    model.compile(loss={'label': 'sparse_categorical_crossentropy',
                        'box': 'mse'},
                  metrics={'label': 'accuracy',
                           'box': mIoU})

    result = model.evaluate(testloader)

    print('mIoU: {:.7}%, classification accuracy: {:.5}%, {} valid'
          .format(result[4] * 100, result[3] * 100, len(valid_filenames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_dir', required=True, type=str,
                        help='Valid dataset directory. Example: D:\cats_dogs_dataset.')
    parser.add_argument('--model_filename', required=True, type=str,
                        help='File name with model to load. Example: simpleCNN_0.h5.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--channels', type=int, default=3)
    args = parser.parse_args()

    evaluate_model(args.valid_dir, args.model_filename,
                   args.batch_size, args.image_size, args.channels)

