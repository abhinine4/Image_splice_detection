import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

class Manip(nn.Module):
    def __init__(self, num_classes):
        super(Manip, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(32,32,kernel_size=5, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(115200, 256)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        # x = nn.functional.softmax(x, dim=1)
        return x
    
if __name__ == '__main__':
    # cudnn.benchmark = True
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = Manip(2).to(device)
    # print(model)
    x = torch.randn(1, 3, 128, 128).to(device)
    out = model(x)
    print(out)
        
