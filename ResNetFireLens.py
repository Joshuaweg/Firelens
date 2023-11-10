
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import rasterio
import os
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import shutil
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
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from collections import OrderedDict
import cv2
import copy

writer = SummaryWriter("runs/test-2")
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 # --- ensures seed in range of 32bit int
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)
### ***************************************************************
# Device config GPU or CPU or mps(MAC)
comp_hardware = 'MAC'
if comp_hardware == 'PC':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # --- PC
    print('Using device:', device)
elif comp_hardware == 'MAC':
    device = torch.device("mps") # --- MAC, use Apple silicon hardware acceleration
    print('Using device:', device)
    
### ***************************************************************

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
### ***************************************************************

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
    
### *********************************************************************
### ******************** Test Net (20 conv layers) **********************
### *********************************************************************
# -------- Residual Block ------------------------
# Basic building block of the residual network
# ---------------------------------------------
class RsBlock(nn.Module):
    ### ---------------------------------------------
    ### 2 x Conv layers
    ### 2 x normalization layers
    ### if downsample used --> 1 x Conv layer
    ###                    --> 1 x normalization layer
    ### downsampling input of one block to match size from output of one block in order to match size to add element wise
    ### ----------------------------------------------
    expansion = 1 # --- num of out channels of the block is same as num of out channels of conv layers within block

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # 3x3 kernel size, 1 stride, 1 padding to maintain spatial resolution
        self.bn1 = nn.BatchNorm2d(out_channels)
        # spatial dimensions remain the same
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None: # out --> in to match in size  
            identity = self.downsample(x)

        out += identity  # Add the input
        out = F.relu(out)
        return out
### ***************************************************************
# -------- Net complete architecture ------------------------
# Complete architecture 
# 1st layer + RsBlocks (with skip and downsample) + Fc
# ------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        # First Layer
        # increase feature maps from 3 -> 64 at first layer, improve feature extraction while
        # maintaining spacial resolution
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        # Residual Blocks
        self.layer1 = self._make_layer(RsBlock, 64, 2, stride=1)  # Use 2 RsBlocks, spatial resolution same as output from first layer after normalization
        self.layer2 = self._make_layer(RsBlock, 128, 2, stride=2) # Use 2 RsBlocks, stride 2 halves spatial resolution to --> 1/2
        self.layer3 = self._make_layer(RsBlock, 256, 2, stride=2) # Use 2 RsBlocks, stride 2 halves spatial resolution again to --> 1/4
        self.layer4 = self._make_layer(RsBlock, 512, 2, stride=2) # Use 2 RsBlocks, stride 2 halves spatial resolution again to --> 1/8
        # Pooling and fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # --- avg pooling
        self.fc = nn.Linear(512 * RsBlock.expansion, 6) # --- fully connected layer

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        # --- for downsampling, use 1x1 kernal conv with stride 2 to reduce the spatial resolution
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential( ## --- subclass of nn.Module (layers are sequentially called on the outputs. no need to define foward pass)
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion) 
            )
        ### Creates sequential RsBlocks
        ### returns 
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    # Definng forward pass
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
### *********************************************************************
### ************************** End Test Net 18 **************************
### *********************************************************************

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    print(img.shape)
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
### ***************************************************************
### Splitting dataset for test and train

path = f"images/data"
sid = SatelliteImageDataset(path,True)
un_sid = SatelliteImageDataset(path,False)
train_size = int(0.8 * len(sid)) # ---- training with 80% of total data
test_size = len(sid) - train_size # ---- testing with 20% of total data
train_dataset, test_dataset = torch.utils.data.random_split(sid, [train_size, test_size])

### ***************************************************************
### ***************************************************************
#### ------------------------------- Save Split Datasets to folder for test later

# Function to save a subset of data to a directory
def save_dataset_subset(dataset_subset, target_dir, original_dir):
    for idx in dataset_subset.indices:
        # Get the filename for the current index
        img_filename = dataset_subset.dataset.img_labels[idx]
        # Define the source path of the image file
        src_file_path = os.path.join(original_dir, img_filename)
        # Define the destination path of the image file
        dst_file_path = os.path.join(target_dir, img_filename)
        # Copy the image from the source to the destination
        shutil.copy(src_file_path, dst_file_path)      
        
train_dir = 'images/training_set'
test_dir = 'images/testing_set'
# Create the directories if they don't exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
# Now save the train and test datasets to their respective directories
save_dataset_subset(train_dataset, train_dir, sid.img_dir)
save_dataset_subset(test_dataset, test_dir, sid.img_dir)

#### ------------------------------- Saved Split Datasets to folder for test later
### ***************************************************************
### ***************************************************************

### *********************** Batch SIZE Test ***********************
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
### ***************************************************************

g.manual_seed(31415)
micro_loader_norm = DataLoader(sid,batch_size =25,shuffle = False)
micro_loader = DataLoader(un_sid,batch_size = 25,shuffle = False)
print(len(sid))
exam = iter(micro_loader)
samps, labs = next(exam)
j = 0
### ***************************************************************

for i in range(25):
    plt.subplot(5,5,j+1)
    plt.imshow(np.transpose(samps[i].cpu(),(1,2,0)))
    j+=1
### ***************************************************************

img_grid = make_grid(samps)
writer.add_image("sat_images",img_grid)
exam = iter(micro_loader_norm)
samps, labs = next(exam)
j =0 
### ***************************************************************

for i in range(25):
    plt.subplot(5,5,j+1)
    plt.imshow(np.transpose(samps[i].cpu(),(1,2,0)))
    j+=1
    
### ***************************************************************

img_grid = make_grid(samps)
writer.add_image("norm_images",img_grid)
net = Net()
# net.to("cuda" if torch.cuda.is_available() else "cpu") 
### ******************************************
net.to(device) 
### ******************************************
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=.0001,weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=10, verbose=True)
epochs = 120
n_total_steps = len(train_loader)
labels1 = []
preds = []
y_true= []
y_pred = []
### ***************************************************************
image_names = []
predictions_list = []
best_accuracy = 0.0 # --- Keeps track of best accuracy
best_model_weights = copy.deepcopy(net.state_dict())  # --- Use deepcopy to save a separate copy of weights for the best model found
path_model_save = 'saved_best_model_weights' # --- dir to folder for model saves
if not os.path.exists(path_model_save): # --- check dir exist or make dir
    os.makedirs(path_model_save)
### ***************************************************************
### ***************************************************************
    
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_correct = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs.to(device)
        labels.to(device)
        inputs, labels = inputs.to(device), labels.to(device)
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
        ### ***********************************************
        # mini-batch contains 32 samples from test dataset
        ### ***********************************************
        if i % 5 == 4:    # print every 5 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
            writer.add_scalar('training loss', running_loss/5, epoch*n_total_steps+i)
            writer.add_scalar('accuracy', running_correct/5, epoch*n_total_steps+i)
            running_loss = 0.0
            running_correct = 0.0
    correct = 0
    total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
    valid_loss = 0.0
    
    ### ***************************************************************

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
            # Store the image names and their predictions for later
            batch_image_names = [labels_map[values_map[l.item()]] for l in labels]  # Convert numeric labels back to string labels
            image_names.extend(batch_image_names)
            img_name_str = [values_map[num] for num in image_names] # convert the image name from num to str
            batch_predictions = [values_map[p.item()] for p in predicted]
            predictions_list.extend(batch_predictions)
            
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
                
### ***************************************************************
                

        epoch_accuracy = 100 * correct / total
        if epoch == (epochs-1):
            preds = torch.cat([torch.stack(batch) for batch in preds])
            labels1 = torch.cat(labels1)
        scheduler.step(valid_loss)
        print(f'Accuracy of the network on the 1440 test images: {epoch_accuracy:.2f}')
     
        ### vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        ### *************************** Best Training Model Save to File *******************************
        ### ********************************************************************************************

        # torch.save(net.state_dict(),"fireRSNet18_test2.pth") ### check the accuracy and save best training model **** !!!
        if epoch_accuracy > best_accuracy: # --- check accuracy for best test accuracy score
            best_accuracy = epoch_accuracy
            best_model_weights = copy.deepcopy(net.state_dict()) # --- make a copy of weights of the best model for calculation use later
            torch.save(best_model_weights, 'saved_best_model_weights/FL_model_best.pth') # --- saving the best model weights to a file
            print("New best model found and saved.")
        ### ********************************************************************************************
        ### *************************** Best Training Model Saved to File ******************************
        ### ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
### ***************************************************************
### Confusion Matrix
net.load_state_dict(best_model_weights) ## --- load best model found weights

# reset 
y_true = []
y_pred = []

with torch.no_grad(): ## -- run model without grad descent
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

cm = confusion_matrix(y_true, y_pred)
print(cm)

df_cm = pd.DataFrame(cm, index=[i for i in values_map.values()],
                     columns=[i for i in values_map.values()])

df_cm2 = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index = [i for i in values_map.values()],
                      columns = [i for i in values_map.values()])

# Plotting standard confusion matrix
plt.figure(figsize=(12,7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.title('STD Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('best_model_SCF_best.png')  # Save the standard confusion matrix plot
plt.show()

# Plotting normalized confusion matrix
plt.figure(figsize = (12,7))
sn.heatmap(df_cm2, annot=True, cmap='Blues', fmt='.2%')
plt.title('Normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('best_model_NCF_best.png')  # Save the normalized confusion matrix plot
plt.show()
#plt.savefig('best_model_conf_mat.png')

print('Finished Training with best model saved.')
### ***************************************************************



            