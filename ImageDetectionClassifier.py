

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

writer = SummaryWriter("runs/test-2")
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)
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
        label = self.img_labels[idx][:self.img_labels[idx].find("_")]
        label = torch.tensor(labels_map[label],dtype=torch.long)
        label.to(device)
        image = ToTensor()(image)
        image = image.permute(1,2,0)
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



def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    print(img.shape)
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
path = f"images/data"
sid = SatelliteImageDataset(path,True)
un_sid = SatelliteImageDataset(path,False)
train_size = int(0.8 * len(sid))
test_size = len(sid) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(sid, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
g.manual_seed(31415)
micro_loader_norm = DataLoader(sid,batch_size =25,shuffle = False)
micro_loader = DataLoader(un_sid,batch_size = 25,shuffle = False)
print(len(sid))
exam = iter(micro_loader)
samps, labs = next(exam)
j = 0
for i in range(25):
    plt.subplot(5,5,j+1)
    plt.imshow(np.transpose(samps[i].cpu(),(1,2,0)))
    j+=1
img_grid = make_grid(samps)
writer.add_image("sat_images",img_grid)
exam = iter(micro_loader_norm)
samps, labs = next(exam)
j =0 
for i in range(25):
    plt.subplot(5,5,j+1)
    plt.imshow(np.transpose(samps[i].cpu(),(1,2,0)))
    j+=1
img_grid = make_grid(samps)
writer.add_image("norm_images",img_grid)
net = Net()
net.to("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=.0001,weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=10, verbose=True)
epochs = 50
n_total_steps = len(train_loader)
labels1 = []
preds = []
y_true= []
y_pred = []
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_correct = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs.to(device)
        labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predictions = torch.max(outputs, 1)
        running_correct +=(predictions == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
            writer.add_scalar('training loss', running_loss/5, epoch*n_total_steps+i)
            writer.add_scalar('accuracy', running_correct/5, epoch*n_total_steps+i)
            running_loss = 0.0
            running_correct = 0.0
    correct = 0
    total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
    valid_loss = 0.0
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data in test_loader:
            images, labels = data
                # calculate outputs by running images through the network
            outputs = net(images)
            
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if epoch == (epochs-1):
                outputs_cm = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
                y_pred.extend(outputs_cm)
            
                y_true.extend(labels.cpu().numpy()) # Save Truth
                n_samples += labels.shape[0]
                n_correct += (predicted == labels).sum().item()
                class_predictions = [F.softmax(output,dim=0) for output in outputs]
                preds.append(class_predictions)
                labels1.append(predicted)
        if epoch == (epochs-1):
            preds = torch.cat([torch.stack(batch) for batch in preds])
            labels1 = torch.cat(labels1)
        scheduler.step(valid_loss)
        print(f'Accuracy of the network on the 1220 test images: {100 * correct // total} %')
classes = range(6)
torch.save(net.state_dict(),"wfclassifer2.pth")
for cl in classes:
    labels_i = labels1 == cl
    pred_i = preds[:,cl]
    writer.add_pr_curve(values_map[cl],labels_i,pred_i,global_step=0)
writer.close()
cm = confusion_matrix(y_true, y_pred)
print(cm)
df_cm = pd.DataFrame(cm/ np.sum(cm, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output_old_2.png')
print('Finished Training')

