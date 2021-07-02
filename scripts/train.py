import ssl
import argparse
from utils import *
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
from os.path import abspath, splitext
from os import listdir

# ssl._create_default_https_context = ssl._create_unverified_context


def create_simpleCNN_model(image_size, channels):
    input = Input(shape=(image_size, image_size, channels))
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    class_prediction = Dense(128, activation='relu')(x)
    class_prediction = Dense(64, activation='relu')(class_prediction)
    class_prediction = Dropout(0.2)(class_prediction)
    class_prediction = Dense(32, activation='relu')(class_prediction)
    class_prediction = Dropout(0.2)(class_prediction)
    class_prediction = Dense(2, activation='softmax', name='label')(class_prediction)

    box_predictions = Dense(128, activation='relu')(x)
    box_predictions = Dense(64, activation='relu')(box_predictions)
    box_predictions = Dropout(0.2)(box_predictions)
    box_predictions = Dense(32, activation='relu')(box_predictions)
    box_predictions = Dropout(0.2)(box_predictions)
    box_predictions = Dense(4, activation='sigmoid', name='box')(box_predictions)

    return Model(inputs=input, outputs=[class_prediction, box_predictions])


def create_NASnetMobile_model(image_size, channels):
    N_mobile = tf.keras.applications.NASNetMobile(input_tensor=
                                                  Input(shape=(image_size, image_size, channels)),
                                                  include_top=False, weights='imagenet')
    N_mobile.trainable = False
    base_model_output = N_mobile.output

    flattened_output = GlobalAveragePooling2D()(base_model_output)

    class_prediction = Dense(256, activation="relu")(flattened_output)
    class_prediction = Dense(128, activation="relu")(class_prediction)
    class_prediction = Dropout(0.2)(class_prediction)
    class_prediction = Dense(64, activation="relu")(class_prediction)
    class_prediction = Dropout(0.2)(class_prediction)
    class_prediction = Dense(32, activation="relu")(class_prediction)
    class_prediction = Dense(2, activation='softmax', name="label")(class_prediction)

    box_output = Dense(256, activation="relu")(flattened_output)
    box_output = Dense(128, activation="relu")(box_output)
    box_output = Dropout(0.2)(box_output)
    box_output = Dense(64, activation="relu")(box_output)
    box_output = Dropout(0.2)(box_output)
    box_output = Dense(32, activation="relu")(box_output)
    box_predictions = Dense(4, activation='sigmoid', name="box")(box_output)

    return Model(inputs=N_mobile.input, outputs=[class_prediction, box_predictions])


def train_model(data_dir, model, image_size, channels, valid_size, batch_size, epochs, save_name, lr,
                es_min_delta, es_patience, rlr_factor, rlr_patience):
    data_dir = abspath(data_dir)

    file_names = list(set([splitext(filename)[0] for filename in listdir(data_dir)]))

    train_file_names, val_file_names = train_test_split(np.array(file_names), test_size=valid_size)

    trainloader = load_dataset(data_dir, train_file_names, batch_size, image_size, channels)
    testloader = load_dataset(data_dir, val_file_names, batch_size, image_size, channels)

    if model:
        model = create_NASnetMobile_model(image_size, channels)
    else:
        model = create_simpleCNN_model(image_size, channels)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss={'label': 'sparse_categorical_crossentropy',
                        'box': 'mse'},
                  loss_weights={'label': 1.0,
                                'box': 1.0},
                  metrics={'label': 'accuracy',
                           'box': mIoU})

    stop = EarlyStopping(monitor="val_loss", min_delta=es_min_delta, patience=es_patience,
                         restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=rlr_factor, patience=rlr_patience,
                                  min_lr=1e-7, verbose=1)

    model.fit(trainloader, validation_data=testloader, epochs=epochs,
              callbacks=[stop, reduce_lr], verbose=1)

    model.save(save_name + '.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, type=str,
                        help='Dataset directory. Example: D:\cats_dogs_dataset.')
    parser.add_argument('--model', type=int, default=1,
                        help='Model to train: 0 - simpleCNN or 1 - NASnetMobile.')
    parser.add_argument('--valid_size', type=float, default=0.1,
                        help='The proportion of the dataset to include in '
                             'the valid split (from 0.0 to 1.0).')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_savename', type=str, default='NASnetMobile_0',
                        help='File name for saving model. Example: simpleCNN_0.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--es_mindelta', type=float, default=1e-4,
                        help='Minimum change in the monitored quantity to qualify '
                             'as an improvement (for EarlyStopping func).')
    parser.add_argument('--es_patience', type=int, default=50,
                        help='Number of epochs with no improvement after which '
                             'training will be stopped (for EarlyStopping func).')
    parser.add_argument('--rlr_factor', type=float, default=0.25,
                        help='Factor by which the learning rate will be reduced '
                             '(for ReduceLROnPlateau func).')
    parser.add_argument('--rlr_patience', type=int, default=15,
                        help='Number of epochs with no improvement after which '
                             'learning rate will be reduced(for ReduceLROnPlateau func).')
    args = parser.parse_args()

    if args.model == 0 or args.model == 1:
        train_model(args.dataset_dir, args.model, args.image_size, args.channels,
                    args.valid_size, args.batch_size, args.epochs, args.model_savename, args.lr,
                    args.es_mindelta, args.es_patience, args.rlr_factor, args.rlr_patience)
    else:
        raise Exception('There is no model with such a name. Try again. '
                        'Available models: 0 - simpleCNN, 1 - NASnetMobile.')
