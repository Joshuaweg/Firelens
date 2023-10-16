from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.utils import make_grid
import rasterio
import os
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
labels_map = {
    "cloud":0,
    "dust":1,
    "haze":2,
    "land":3,
    "seaside":4,
    "smoke":5
}
values_map = {
    0:"cloud",
    1:"dust",
    2:"haze",
    3:"land",
    4:"seaside",
    5:"smoke"
}
classes = [
    "cloud",
    "dust",
    "haze",
    "land",
    "seaside",
    "smoke"
]

class SatelliteImageDataset(Dataset):
    def __init__(self, img_dir, transform = False):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = os.listdir(img_dir)


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = rasterio.open(img_path)
        image = image.read()
        label =""
        if "Haze" in self.img_labels[idx]:
            label = "haze"
        else:
            label = self.img_labels[idx][:self.img_labels[idx].find("_")].lower()
        label = torch.tensor(labels_map[label],dtype=torch.long)
        label.to(device)
        image = ToTensor()(image)
        image = image[:,:3,:]
        image = image.permute(1,2,0)
        print(self.img_labels[idx])
        print(image.shape)
        print(image)
        if(self.transform):
            image = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        image.to(device)
        return image.to(device), label.to(device)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 256, 1)
        self.fc1 = nn.Linear(246016, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
criterion = nn.CrossEntropyLoss()
path = f"images/new"
sid = SatelliteImageDataset(path,True)
test_loader = DataLoader(sid, batch_size=32, shuffle=False)
net = Net()
net.load_state_dict(torch.load("wfclassifer2.pth"))
net.eval()
net.to("cuda" if torch.cuda.is_available() else "cpu")
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
valid_loss = 0.0
epochs = 1
preds = []
labels1 = []
y_pred = []
y_true = []

for epoch in range(epochs):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data in test_loader:
            images, labels = data
                # calculate outputs by running images through the network
            outputs = net(images)
            outputs_cm = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(outputs_cm)
            
            y_true.extend(labels.cpu().numpy()) # Save Truth
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if epoch == (epochs-1):
                n_samples += labels.shape[0]
                n_correct += (predicted == labels).sum().item()
                class_predictions = [F.softmax(output,dim=0) for output in outputs]
                preds.append(class_predictions)
                labels1.append(labels)
        if epoch == (epochs-1):
            preds = torch.cat([torch.stack(batch) for batch in preds])
            labels1 = torch.cat(labels1)
print(y_true,y_pred)
print(f'Accuracy of the network on the 1220 test images: {100 * correct // total} %')
cm = confusion_matrix(y_true, y_pred)
print(cm)
df_cm = pd.DataFrame(cm/ np.sum(cm, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output4.png')