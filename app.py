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
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import LRP
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule

writer = SummaryWriter("runs/test-2")
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 # --- ensures seed in range of 32bit int
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)
### ***************************************************************
# Device config GPU or CPU or mps(MAC)
comp_hardware = 'PC'
if comp_hardware == 'PC':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # --- PC
    #device = torch.device('cpu')
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
        image = image.read([1,2,3])
        name = self.img_labels[idx]
        label = self.img_labels[idx][:self.img_labels[idx].find("_")]
        label = torch.tensor(labels_map[label.lower()],dtype=torch.long)
        label.to(device)
        image = ToTensor()(image)
        image = image.permute(1,2,0)
        if(self.transform):
            image = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        image.to(device)
        return image.to(device), label.to(device),name
    
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
        self.layer2 = self._make_layer(RsBlock, 64, 2, stride=2) # Use 2 RsBlocks, stride 2 halves spatial resolution to --> 1/2
        self.layer3 = self._make_layer(RsBlock, 128, 2, stride=2) # Use 2 RsBlocks, stride 2 halves spatial resolution again to --> 1/4
        self.layer4 = self._make_layer(RsBlock, 128, 2, stride=2) # Use 2 RsBlocks, stride 2 halves spatial resolution again to --> 1/8
        self.layer5 = self._make_layer(RsBlock, 128, 2, stride=2)
        # Pooling and fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # --- avg pooling
        self.fc = nn.Linear(128 * RsBlock.expansion, 6) # --- fully connected layer

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
        x = self.layer5(x)
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
    
#initialize model
net = Net()
net.load_state_dict(torch.load('saved_best_model_weights/FL_Model_best4.pth',map_location=device)) ## --- load best model found weights
net.eval()
net.to(device)
# 
integrated_gradients = IntegratedGradients(net)
path="read_in"
img1 = rasterio.open( path+'/'+os.listdir(path)[0])
print("Reading file: ",os.listdir(path)[0])
img1 = img1.read([1,2,3])
img1= ToTensor()(img1).to(device)
img1 = img1.permute(1,2,0)
img1_trans = Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(img1)
img1_trans=img1_trans.unsqueeze(0)
img1_fd = img1.unsqueeze(0)
img1_trans = img1_trans.reshape(1, 3, 256, 256)
v_outputs = net(img1_trans)
_,v_predicted = torch.max(v_outputs, 1)
print("Classified Areosol in Image: ",values_map[v_predicted[0].item()])
print("generating Integrated Gradients Activation Map...")
attributions_ig = integrated_gradients.attribute(img1_fd, target=v_predicted[0], n_steps=50)
        #attributions_ig_nt = noise_tunnel.attribute(img1_trans, nt_samples=10, nt_type='vargrad', target=v_predicted[0])
        # attributions_lrp = lrp.attribute(img1_trans, target=v_predicted[0])
        # attributions_gs = gradient_shap.attribute(img1_trans,
        #                                   n_samples=50,
        #                                   stdevs=0.0001,
        #                                   baselines=rand_img_dist,
        #                                   target=predicted[0])
default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

_ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(img1.squeeze().cpu().detach().numpy(), (1,2,0)),
                             ["original_image", "heat_map"],
                             ["all", "positive"],
                             cmap=default_cmap,
                             show_colorbar=True,
                             outlier_perc = 2
)
