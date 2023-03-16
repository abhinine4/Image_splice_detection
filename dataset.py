import glob
import os
import io
import torch
import random
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import PIL.Image as pil
from torchvision import transforms
from PIL import Image, ImageChops, ImageEnhance
# from keras.utils.np_utils import to_categorical
import torch.nn.functional as F

class ManipData(DataLoader):
    def __init__(self, image_dir, mode):
        self.image_dir = image_dir
        jpg_ds = glob.glob(self.image_dir+'/*/*jp*g')
        tif_ds = glob.glob(self.image_dir+'/*/*tif')
        bmp = glob.glob(self.image_dir+'/*/*bmp')
        self.data_ds = jpg_ds + tif_ds + bmp
        random.shuffle(self.data_ds)
        if mode == 'train':
            self.model_data = self.data_ds[:len(self.data_ds)-1000]
        elif mode == 'val':
            self.model_data = self.data_ds[len(self.data_ds)-1000:]

    def cv2_enhance_contrast(self, img, factor):
        mean = np.uint8(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0])
        img_deg = np.ones_like(img) * mean
        return cv2.addWeighted(img, factor, img_deg, 1-factor, 0.0)

    def ela(self,path):
        quality = 90
        scale = 15
        orig = cv2.imread(path)
        orig = cv2.resize(orig, (128, 128), interpolation = cv2.INTER_AREA)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        _, buffer = cv2.imencode(".jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed_img = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)

        diff = scale * (cv2.absdiff(orig, compressed_img))
        ediff = self.cv2_enhance_contrast(diff, 0.25)
        ediff = ediff.astype(float)/255.0
        orig = orig.astype(float)/255.0
        return ediff, orig
    
    def __getitem__(self, idx):
        file_path = self.model_data[idx]
        parts = file_path.split('/')
        label = 1 if parts[-2] == 'Au' else 0
        label = torch.tensor(label)
        label = F.one_hot(label, num_classes=2)
        ela, orig = self.ela(file_path)
        return ela, label, orig

    def __len__(self):
        return len(self.model_data)

if __name__ == '__main__':
    path = './datasets/detection/CASIA2'
    
    manipdataset = ManipData(path, mode='train')
    dataloader = DataLoader(dataset=manipdataset,
                            batch_size=32, 
                            shuffle=True, 
                            num_workers=8, 
                            pin_memory=True,
                            drop_last=True)
    for data in dataloader:
        img, label, orig = data
        img = img.permute(0,3,1,2)
        orig = orig.permute(0,3,1,2)
        break
        
    




