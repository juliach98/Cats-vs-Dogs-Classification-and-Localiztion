import cv2
from os import listdir, mkdir
from os.path import splitext, abspath, join, exists
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import argparse


class DataAugmentation(object):
    def __init__(self, data_directory, aug_directory, multiple):
        self.data_directory = abspath(data_directory)
        self.aug_directory = abspath(aug_directory)
        self.multiple = multiple

        self.image_augmentations = iaa.SomeOf((1, 3), [
            iaa.GaussianBlur(sigma=(0, 2.0)),
            iaa.Sometimes(0.5, iaa.Affine(translate_percent={"x": (-20, 20), "y": (-20, 20)}, mode='reflect')),
            iaa.Affine(rotate=(-20, 20), mode='reflect'),

            iaa.AdditiveGaussianNoise(scale=(0, 0.07 * 255)),

            # Flip/mirror image horizontally
            iaa.Fliplr(0.5),

            # Flip/mirror image vertically
            iaa.Flipud(0.3),

            # Rotate image by 90 or 270 degrees
            iaa.Rot90([1, 3]),

            # Increase or decrease the brightness
            iaa.Multiply((0.5, 1.5)),

            iaa.Sometimes(0.5, iaa.MotionBlur(k=5))
        ])

    def augmentation(self, image, box):
        # Give the bounding box to imgaug library
        bounding_box = BoundingBoxesOnImage.from_xyxy_array(box, shape=image.shape)

        # Perform random augmentations
        image_aug, box_aug = self.image_augmentations(image=image, bounding_boxes=bounding_box)

        return image_aug, box_aug

    def image_aug(self):
        if not exists(self.aug_directory):
            mkdir(self.aug_directory)

        filenames = list(set([splitext(filename)[0] for filename in listdir(self.data_directory)]))

        for i in range(self.multiple):
            # Postfix to the each different augmentation of one image
            image_postfix = str(1000 + i)

            for filename in filenames:
                image = cv2.imread(join(self.data_directory, filename + '.jpg'))

                with open(join(self.data_directory, filename + '.txt'), 'r') as txt:
                    line = txt.readline().split()
                    label = line[0]
                    box = [[int(x) for x in line[1:]]]

                aug_image, aug_box = self.augmentation(image, box)

                # Discard the the bounding box going out the image completely
                aug_box = aug_box.remove_out_of_image()

                # Clip the bounding box that are only partially out of the image
                aug_box = aug_box.clip_out_of_image()

                # Get rid of the the image if bounding box was discarded
                if len(aug_box.bounding_boxes) == 1:
                    cv2.imwrite(join(self.aug_directory, filename + image_postfix + '.jpg'), aug_image)

                    with open(join(self.aug_directory, filename + image_postfix + '.txt'), 'w') as txt:
                        txt.write(label + ' ' + str(int(aug_box.items[0].x1)) + ' ' + str(int(aug_box.items[0].y1)) +
                                  ' ' + str(int(aug_box.items[0].x2)) + ' ' + str(int(aug_box.items[0].y2)) + ' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, type=str,
                        help='Dataset directory. Example: D:\cats_dogs_dataset.')
    parser.add_argument('--aug_dir', required=True, type=str,
                        help='Augmentation dataset directory. Example: D:\cats_dogs_dataset_aug.')
    parser.add_argument('--mult', type=int, default=8,
                        help='How many times you need to increase the dataset. Example: 8.')
    args = parser.parse_args()

    data_aug = DataAugmentation(args.dataset_dir, args.aug_dir, args.mult)
    data_aug.image_aug()
