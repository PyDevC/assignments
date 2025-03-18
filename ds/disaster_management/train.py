import torch
from torchvision import transforms 
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

img_augmentation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])       
    ])

class LouDataset(Dataset):

    def __init__(self,type, img_path):
        self.img_id = pd.read_csv(img_path+f'{type}.csv')
        self.img_folder = img_path+f'{type}/'
        self.imgs = self.df[self.df['Normal']==1]
        self.imgs = self.imgs['Image ID'].values


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self):
        before_img_path = os.path.join(self.img_folder, self.imgs[idx])
        flood_img_path = glob(self.img_folder+self.imgs[idx].split('.')[0]+'_*')[0]

        before_img = Image.open(before_img_path)
        before_img = img_augmentation(before_img)
        flood_img = Image.open(flood_img_path)
        flood_img = img_augmentation(flood_img)

        label = int(flood_img_path.split('_')[-1].split('.')[0])
        
        return (before_img,flood_img,torch.as_tensor([label], dtype=torch.float))


train = LouDataset('train','main path')
test = LouDataset('test','')


# loading the dataset for model creation
