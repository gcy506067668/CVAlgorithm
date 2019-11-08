import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    classes = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
        for cl in _:
            classes.append(cl.split('/')[-1])

    return images, classes

if __name__ =='__main__':
    dir = '/home/zjq/plate_classification/transfer_learning_resnet18-master/dataset'
    image_path,image_class = make_dataset(dir)
    image_path = sorted(image_path)
    for i in image_path:
        print(i)
        x = Image.open(i).convert('RGB')



