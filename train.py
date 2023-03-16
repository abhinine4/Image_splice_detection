import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset import ManipData
from model import Manip
import copy
import pickle
from datetime import datetime
from config import *
import argparse

cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, help='Batch size for data loader', default=64)
    parser.add_argument('--epoch', type=int, help='max epochs', default=30)
    parser.add_argument('--classes', type=int, help='number of classes', default=2)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0005)
    parser.add_argument('--eps', type=float, help='epsilon value for RMSProp', default=1e-08)
    parser.add_argument('--decay', type=float, help='weight decay for RMSProp', default=0.0)
    parser.add_argument('--alpha', type=float, help='alpha value for RMSProp', default=0.9)
    return parser.parse_args()

def train(model, train_dl, criterion, optimizer, device):
    
    train_loss = 0.0
    for data in tqdm(train_dl):
        ela, label, orig = data
        ela = ela.permute(0,3,1,2)
        ela = ela.to(device)

        label = label.float()
        label = label.to(device)
        
        output = model(ela).to(device)
        model.train()
        
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item() * ela.size(0)
    train_loss /= len(train_dl.dataset)
    return train_loss

def eval(model, test_dl, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data in test_dl:
            ela, label, orig = data
            ela = ela.permute(0,3,1,2)
            ela = ela.to(device)
            
            label = label.float()
            label = label.to(device)

            output = model(ela).to(device)
            loss = criterion(output, label)
            test_loss += loss.item() * ela.size(0)

            output = torch.sigmoid(output)

            output[output >= 0.5] = 1
            output[output < 0.5] = 0 

            correct += torch.sum(torch.argmax(output,1) == torch.argmax(label,1)).item()

    test_loss /= len(test_dl.dataset)
    accuracy = 100*(correct/len(test_dl.dataset))
    return test_loss, accuracy

def get_dataloader(batch_size=None,mode=None):
    dataset = ManipData(image_dir,mode)
    dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=8, 
                                pin_memory=True,
                                drop_last=True)
    return dataloader

    
if __name__ == '__main__':
    args = get_args()

    best_acc = 0
    tr_loss = []
    val_loss = []
    model = Manip(num_classes=args.classes).double().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters() ,lr=args.lr, alpha=args.alpha, eps=args.eps, weight_decay=args.decay)
    
    train_dataloader = get_dataloader(batch_size=args.batch, mode='train')
    eval_dataloader = get_dataloader(batch_size=args.batch, mode='val')

    for epoch in range(args.epoch):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        eval_loss, accuracy = eval(model, eval_dataloader, criterion, device)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Validation Loss: {eval_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        tr_loss.append(train_loss)
        val_loss.append(eval_loss)

        if accuracy > best_acc:
            best_acc = accuracy
            best_weights = copy.deepcopy(model.state_dict())

    model_ts = Manip(num_classes=args.classes).double().to(device)
    model_ts.load_state_dict(best_weights)

    # for params in model_ts.parameters():
    #     print(params)

    torch.save(model_ts,os.path.join(save_dir, 'mode_weights_{}.pth'.format(datetime.now().strftime("%Y%m%d_%H"))))
    with open(save_dir+'train_loss_{}.pkl'.format(datetime.now().strftime("%Y%m%d_%H")), "wb") as fp:   
        pickle.dump(tr_loss, fp)

    with open(save_dir+'validation_loss_{}.pkl'.format(datetime.now().strftime("%Y%m%d_%H")), "wb") as fp:   
        pickle.dump(val_loss, fp)

    print("Training completed. Best Validation accuracy : {}.  Weights saved at {}.".format(best_acc, save_dir))



