import os
import torch
import cv2
import glob
import random
import PIL.Image as pil
import numpy as np
import torch.nn as nn
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Prediction:
    def __init__(self,model_path, dir):
        self.model_path  = model_path
        self.dir = dir
                
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

    def get_image(self):
        jpg_ds = glob.glob(self.dir+'/*/*jp*g')
        tif_ds = glob.glob(self.dir+'/*/*tif')
        bmp = glob.glob(self.dir+'/*/*bmp')
        data_ds = jpg_ds + tif_ds + bmp
        data_ds.sort(key=len)
        print(data_ds)
        random.shuffle(data_ds)
        selected = random.sample(data_ds, 200)
        return selected
    
    def predict_manip(self):
        model = torch.load(self.model_path,map_location=device)
        model.eval()
        # for params in model.parameters():
        #      print(params)

        image_paths = self.get_image()
        c = 0
        cor = 0
        for pth in image_paths:
            print(pth)
            ela_image, orig_image = self.ela(pth)
            parts = pth.split('/')
            orig_label = "Authentic" if parts[-2] == 'Au' else "Manipulated"
            ela_image = torch.from_numpy(ela_image)
            ela_image = ela_image.unsqueeze(0)
            ela_image = ela_image.permute(0,3,1,2).to(device)
            

            prediction = model(ela_image)

            prediction = torch.sigmoid(prediction)
            print(prediction)
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            predicted = torch.argmax(prediction,1)
            pred = "Authentic" if predicted[0] == 1 else "Manipulated"
            if orig_label == 'Authentic':
                cor += 1
            if pred == orig_label:
                c += 1

        print("\nTotal authentic : {}".format(cor))
        print("Total manipultaed : {}".format(len(image_paths)-cor))
        return  c/len(image_paths)*100
    
if __name__ == '__main__':
    path = './datasets/detection/splice_eval'
    model = '/home/csgrad/akumar58/EVAL4/cnn_manip/train_files/mode_weights_20230307_23.pth'
    prd = Prediction(model, path)
    print("\nAccuracy on cross test dataset : {}%".format(prd.predict_manip()))
    
    


