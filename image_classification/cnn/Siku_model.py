import os
from cnn_finetune import make_model
import torch
import torch.nn as nn
import pretrainedmodels
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from data.plate_dataset import PlateDataset,SinglePlateDataset
import torch.nn.functional as F
import time

#long running
#do something other



def resnext101_32x4d():

    my_resnext = make_model('resnext101_32x4d', num_classes=2, pretrained=True)

    return my_resnext
def test_model(my_model,device,data_loaders,dataset_size):
            score = 0
            my_model.eval()
            for samples in data_loaders:
                inputs = samples['image'].to(device)
                outputs = my_model(inputs)
                epoch_score = F.softmax(outputs,dim = 1).cpu().detach().numpy()
                score += epoch_score.sum(axis=0)[0]
            print("finished predicting")
            return score/dataset_size
class Siku_model():
    def __init__(self):
        self.model = None
        self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),

                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.kwargs =  {'num_workers':4, 'pin_memory':True}
    def load_model(self,path,gpu_ids='-1'):
        try:
            load_path = path
            print('loading the model from %s' % load_path)
            pretrained_net = torch.load(load_path)
            self.device = torch.device('cuda:{}'.format('0') if gpu_ids == '0' else 'cpu')
            my_model = resnext101_32x4d()
            # load the pretrained weight
            pre_dict = my_model.state_dict()
            pretrained_net2 = {k.replace('module.', ''): v for k, v in pretrained_net.items()}
            pretrained_dict = {k: v for k, v in pretrained_net2.items() if k in pre_dict}
            my_model.load_state_dict(pretrained_dict)
            if gpu_ids=='0':
                my_model = nn.DataParallel(my_model).cuda()
            my_model = my_model.to(self.device)
            self.model = my_model
            return True
        except:
            print("Error happend when load model")
            return False
    def test(self,dataroot):
        score = -1
        if(os.path.isfile(dataroot)):
            image_datasets = SinglePlateDataset(root=dataroot, transform = self.transform)
            data_loaders =  DataLoader(image_datasets, batch_size=40, shuffle=False, **self.kwargs)
            dataset_size = len(image_datasets)
            print('one image')
            score = test_model(self.model,self.device,data_loaders,dataset_size)
            return score

        elif(os.path.isdir(dataroot)):
            image_datasets = PlateDataset(root=dataroot, transform = self.transform)
            data_loaders =  DataLoader(image_datasets, batch_size=40, shuffle=False, **self.kwargs)
            dataset_size = len(image_datasets)
            print('image size %s \n'%dataset_size)
            score = test_model(self.model,self.device,data_loaders,dataset_size)
            return score

        else:
            print("Neither file or dir")

            return score
if __name__ == "__main__":
    model = Siku_model()
    if(model.load_model('/home/zjq/lab/LV/checkpoints/pretrained/0621_1157.pth','0')):
       score = model.test('/home/zjq/lab/LV/dataset/2019_6_28_together_4/test/GEN/20190717100544984.jpg')
       print(score)
