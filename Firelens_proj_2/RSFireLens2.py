
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
from torch.cuda.amp import GradScaler, autocast # for pytorch automatic mixed precision for lower memory use (CUDA)
from tqdm import tqdm # for displaying progress bar 

writer = SummaryWriter("runs/test-2") ## --- log for tensorboard 
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 # --- ensures seed in range of 32bit int
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)
### ***************************************************************
# Device config GPU or CPU or mps(MAC)
# Find and use the best configuration for device (CUDA >> MPS >> CPU)

if torch.cuda.is_available(): # --- CUDA
    device = torch.device('cuda')
    print('Using CUDA device:', torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():  # --- MAC
    device = torch.device("mps")
    print('Using MPS device (Apple Silicon)')
else:
    device = torch.device('cpu')  # Fallback to CPU
    print('Using CPU')

print('Selected device:', device)
    
### ***************************************************************
# Transformations and Label Mappings
transform = Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
labels_map = {"cloud": 0, "dust": 1, "haze": 2, "land": 3, "seaside": 4, "smoke": 5}
values_map = {0: "cloud", 1: "dust", 2: "haze", 3: "land", 4: "seaside", 5: "smoke"}

### ***************************************************************
# --- Modified to include finding files in subfolders
class SatelliteImageDataset(Dataset):
    def __init__(self, img_dir, transform=False):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        
        # --- include sub-folders in dir if i want to add extra imgae datas in folder form 
        # --- (easier to idendify the groups of those datas)
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.endswith(('.tif')):  # --- files with tif extension
                    self.img_labels.append(os.path.join(root, file))
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels[idx]
        image = rasterio.open(img_path)
        image = image.read()

        # Extract label from the file name
        label = os.path.basename(img_path).split("_")[0]
        label = torch.tensor(labels_map[label], dtype=torch.long)
        label = label.to(device)

        # Image transformations
        image = ToTensor()(image)
        image = image.permute(1, 2, 0)
        if self.transform:
            image = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

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
        self.layer1 = self._make_layer(RsBlock, 64, 2, stride=1)  # 2 RsBlocks, spatial resolution = 256 x 256, feature channels = 64
        self.layer2 = self._make_layer(RsBlock, 128, 2, stride=2) # 2 RsBlocks, spatial resolution = 128 x 128, feature channels = 128
        self.layer3 = self._make_layer(RsBlock, 256, 2, stride=2) # 2 RsBlocks, spatial resolution = 64 x 64, feature channels = 256
        self.layer4 = self._make_layer(RsBlock, 512, 2, stride=2) # 2 RsBlocks, spatial resolution = 32 x 32, feature channels = 512
        
        ### *****************************
        # Testing with 5 layers ### --- comment out to reduce to 4
        #self.layer5 = self._make_layer(RsBlock, 1024, 2, stride=2) # 2 RsBlocks, spatial resolution = 16 x 16, feature channels = 1024
        ### *****************************

        
        # Pooling and fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # --- avg pooling 1 x 1 x (512 for 4 layers) or (1024 for 5 layers)
        ### *****************************
        ### --- Testing for extra layer // reverse comment to reduce back to 4 and comment out the 5 layer
        self.fc = nn.Linear(512 * RsBlock.expansion, 6) # ---- 4 layers fc
        #self.fc = nn.Linear(1024 * RsBlock.expansion, 6) # ---- 5 layers fc
        ### *****************************
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
        ### *****************************
        #x = self.layer5(x) ### --- testing with 5 layers // comment out to reduce to 4
        ### *****************************
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

### *********************** un-comment this for random split data and comment out the manual split data
# train_size = int(0.8 * len(sid)) # ---- training with 80% of total data
# test_size = len(sid) - train_size # ---- testing with 20% of total data
# train_dataset, test_dataset = torch.utils.data.random_split(sid, [train_size, test_size])
### ***********************

### ***************************************************************
### ***************************************************************
### Custom splitting dataset for train with (74% original + their augmented version data) ~80%
### Custom splitting dataset for test with (26% from original with no augmented version trained + newly collected data) ~20%
### Custom split to ensure no leakage of information between training and testing, so testing data is 100% unseen

## --- save paths for train test data
path_train = f"images/training_data" 
path_test = f"images/testing_data"

#--- transform, normalize data 
train_dataset = SatelliteImageDataset(path_train,True) 
test_dataset = SatelliteImageDataset(path_test,True)

### *********************** Batch SIZE Test ***********************
# --- load data
train_loader = DataLoader(train_dataset, batch_size=26, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

### ***************************************************************

g.manual_seed(31415)
micro_loader_norm = DataLoader(sid,batch_size =26,shuffle = False)
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

### ******************************************
net.to(device) 
### ******************************************
# Hyperparameters config
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=.00008,weight_decay=1e-4) ### -- lr = 0.0001 for batch32, lr = 0.00007 for batch25,26 --> 89.96% acu
scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=8, verbose=True) ### --- patience = 10 for 32, 8 for 25,26
epochs = 200 ### -- batch 32 plateaus at around 100 seems like, 

### ******************************************

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
if device.type == 'cuda':
    scaler = GradScaler()   
     
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_correct = 0.0
    # Wrap dataLoader with tqdm for progress bar
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    
    for i, data in enumerate(train_loader_tqdm, 0): # --- training from train_loader set
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs.to(device)
        labels.to(device)
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        ### *************************
        ### --- with pytorch AMP
        if device.type == 'cuda':
            # Use AMP when running on CUDA
            with autocast():
                outputs = net(inputs)
                _, predictions = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization with scaled loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        ### ***********
        else:
            # Standard execution without AMP (MAC: MPS)
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    
            

        running_correct +=(predictions == labels).sum().item()
        
        ### *************************
        
        # print statistics
        running_loss += loss.item()
        train_loader_tqdm.set_description(f"Epoch {epoch+1}/{epochs} Loss: {running_loss / (i+1):.3f}")
        ### -------------- test device
        #print('Using device:', device)
        ### ---------------
        ### ***********************************************
        # mini-batch contains 32 samples from test dataset
        ### ***********************************************
        if (i+1) % 5 == 0:    
            
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
        for data in test_loader: # --- test on test_loader set
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
        if epoch == (epochs-1): # --- checking the last epoch
            preds = torch.cat([torch.stack(batch) for batch in preds])
            labels1 = torch.cat(labels1)
        scheduler.step(valid_loss)
        print(f'Accuracy of the network on test images: {epoch_accuracy:.2f}')
        
        ### vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        ### *************************** Best Training Model Save to File *******************************
        ### ********************************************************************************************

        if epoch_accuracy > best_accuracy: # --- check accuracy for best test accuracy score
            best_accuracy = epoch_accuracy
            best_epoch = epoch
            best_model_weights = copy.deepcopy(net.state_dict()) # --- make a copy of weights of the best model for calculation use later
            torch.save(best_model_weights, 'saved_best_model_weights/FL_model_best_11_23_2023.pth') # --- saving the best model weights to a file
            print("New best model found and saved.")
        ### ********************************************************************************************
        ### *************************** Best Training Model Saved to File ******************************
        ### ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        print(f'Best Accuracy so far: {best_accuracy:.2f} (at epoch: {best_epoch+1})')
        print()
### ***************************************************************
### Confusion Matrix generation varification

net.load_state_dict(best_model_weights) ## --- load best model found weights + biases

# reset 
y_true = []
y_pred = []

with torch.no_grad(): ## -- run model
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
plt.savefig('best_model_SCF_best_11_23_2023.png')  # Save the standard confusion matrix plot
plt.show()

# Plotting normalized confusion matrix
plt.figure(figsize = (12,7))
sn.heatmap(df_cm2, annot=True, cmap='Blues', fmt='.2%')
plt.title('Normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('best_model_NCF_best_11_23_2023.png')  # Save the normalized confusion matrix plot
plt.show()
#plt.savefig('best_model_conf_mat.png')

print('Finished Training with best model saved.')
### ***************************************************************



            