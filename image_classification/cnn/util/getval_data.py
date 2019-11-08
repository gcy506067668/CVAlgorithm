#split the val set from training
from data.plate_dataset import PlateDataset,ImageFolerSplitter
path ='/home/zjq/lab/LV/dataset/bag/'
ImageFolerSplitter(path =os.path.join(path, 'train') ,test_size =5000)
