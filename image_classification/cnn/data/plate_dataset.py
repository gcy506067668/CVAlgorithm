import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    classes = os.listdir(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
        # for cl in _:
        #     classes.append(cl.split('/')[-1])

    return images, classes


class ImageFolerSplitter():
    def __init__(self, path, test_size=0):
        self.root_dir = path
        self.test_size = test_size
        self.image_path, self.classes = make_dataset(self.root_dir)
        self.image_path = sorted(self.image_path)
        self.x_train, self.x_val = train_test_split(self.image_path, shuffle=True, test_size=self.test_size)
        self.val_path = self.root_dir.replace('/train', '/val')
        for i in self.classes:
            path = os.path.join(self.val_path, i)
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

        for x in self.x_val:
            image = Image.open(x).convert('RGB')
            x_original = x
            x = x.replace('/train', '/val')
            image.save(x)

            os.remove(x_original)


class PlateDataset(Dataset):
    def __init__(self, root, transform=None):
        # def __init__(self, transform=None, image_path=None):
        """
        Args:
        root_dir (string): Directory with all the images
        transform (): Optional transform to be applied on a sample

        """

        self.root_dir = root
        self.transform = transform

        self.image_path, self.classes = make_dataset(self.root_dir)

        self.image_path = sorted(self.image_path)

        self.labels = {'GEN': 0, 'FAKE': 1}

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_p = self.image_path[idx]
        try:
            cvimg = cv2.imread(image_p)
            cv2.normalize(cvimg, cvimg, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # 公式
            image = Image.fromarray(cvimg).convert('RGB')
            # image = Image.open(image_p).convert('RGB')
        except IOError:
            print('cannot convert', image_p)
        if (self.transform != None):
            image = self.transform(image)
        label = 2

        for k in self.labels.keys():
            if k in image_p:
                label = self.labels[k]
                # print('labels:%s path:%s'%(label,image_p))

        return {'image': image, 'label': label, 'path': image_p}


class SinglePlateDataset(Dataset):
    def __init__(self, root, transform=None):
        # def __init__(self, transform=None, image_path=None):
        """
        Args:
        root_dir (string): Directory with all the images
        transform (): Optional transform to be applied on a sample

        """

        self.root_dir = root
        self.transform = transform
        # self.image_path,self.classes = make_dataset(self.root_dir)
        self.image_path = [self.root_dir]
        # self.image_path = sorted(self.image_path)

        self.labels = {'GEN': 0, 'FAKE': 1}

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_p = self.image_path[idx]
        try:
            image = Image.open(image_p).convert('RGB')
        except IOError:
            print('cannot convert', image_p)
        if (self.transform != None):
            image = self.transform(image)
        label = 2

        for k in self.labels.keys():
            if k in image_p:
                label = self.labels[k]
                # print("label:%s path:%s"%(label,path))

        return {'image': image, 'label': label, 'path': image_p}
