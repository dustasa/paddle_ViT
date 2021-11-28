import paddle
import paddle.vision.transforms as T
import numpy as np
import os
import math
from PIL import Image
from paddle.io import Dataset, DataLoader, DistributedBatchSampler
from paddle.vision import transforms, datasets, image_load
from auto_augment import auto_augment_policy_original
from auto_augment import AutoAugment
from random_erasing import RandomErasing


class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return T.resize(image, self.size)


class CenterCrop():
    def __init__(self, size):
        self.size = size

    # image: PIL.image
    def __call__(self, image):
        w, h = image.size
        ch, cw = self.size
        crop_top = int(round(h - ch) / 2.)
        crop_left = int(round(w - cw) / 2.)
        return crop(image, (crop_top, crop_left, ch, cw))


def crop(image, region):
    # region :[x, y ,h , w]
    cropped_image = T.crop(image, *region)
    return cropped_image


class ToTensor():
    def __init__(self):
        pass

    # image: PIL.image
    def __call__(self, image):
        img = paddle.to_tensor(np.array(image))
        if img.dtype == paddle.uint8:
            img = paddle.cast(img, dtype='float32') / 255.
        img = img.transpose([2, 0, 1])
        return img


class Compose():
    def __init__(self, trans):
        self.transforms = trans

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image
        # print("__call__ in Compose")


def main():
    img = Image.open('img.jpg')
    trans = Compose([Resize([256, 256]),
                     CenterCrop([112, 112]),
                     ToTensor()])
    out = trans(img)
    print(out)
    print(out.shape)


if __name__ == "__main__":
    main()
