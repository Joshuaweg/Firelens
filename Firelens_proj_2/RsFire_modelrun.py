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
    worker_seed = torch.initial_seed() % 2**32 
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
### ***************************************************************### ***************************************************************

### ***************************************************************### ***************************************************************

### ***************************************************************### ***************************************************************


class GradCam:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.gradient = None
        self.activation = None
        self.model.eval()
        self.register_hooks()

    def save_gradient(self, grad):
        self.gradient = grad

    def register_hooks(self):
        module = dict([*self.model.named_modules()])[self.layer_name]

        module.register_forward_hook(lambda module, input, output: setattr(self, 'activation', output))
        module.register_backward_hook(lambda module, grad_input, grad_output: self.save_gradient(grad_output[0]))

    def generate_cam(self, input_image, target_class):
        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = np.argmax(output.cpu().data.numpy())

        # Zero grads
        self.model.zero_grad()

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Backward pass
        output.backward(gradient=one_hot_output.to(device), retain_graph=True)

        # Generate CAM
        gradients = self.gradient.cpu().data.numpy()[0]
        activations = self.activation.cpu().data.numpy()[0]
        
        # Average the gradients over the height and width dimensions
        pooled_gradients = np.mean(gradients, axis=(1, 2))
        
        # Multiply each channel by its corresponding gradient
        cam = np.dot(activations.T, pooled_gradients).T

        # Relu on top of the CAM
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_image.shape[2:], interpolation=cv2.INTER_LINEAR)   
        
    
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        # Apply Gaussian Blurring to the CAM
        cam = cv2.GaussianBlur(cam, (5, 5), 0)
        
        return cam



### ***************************************************************### ***************************************************************

def create_custom_colormap():
    # Create a colormap that transitions from blue (less important) to red (more important)
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        colormap[i] = [i, 0, 255 - i]  # Red to Blue
    return colormap

def apply_custom_colormap(heatmap, colormap):
    heatmap_normalized = np.uint8(255 * heatmap)
    colored_heatmap = colormap[heatmap_normalized]
    return colored_heatmap

def apply_heatmap(img, heatmap, alpha=0.4):
    # Generate colormap (red blue)
    colormap = create_custom_colormap()

    # Apply colormap to heatmap
    colored_heatmap = apply_custom_colormap(heatmap, colormap)
    colored_heatmap = np.float32(colored_heatmap) / 255

    # Overlay heatmap on image
    overlayed_img = colored_heatmap * alpha + img * (1 - alpha)
    overlayed_img = np.clip(overlayed_img, 0, 1)
    return overlayed_img
### ***************************************************************### ***************************************************************

### ***************************************************************### ***************************************************************

class SatelliteImageDataset(Dataset):
    def __init__(self, img_dir, transform = False):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = [file for file in os.listdir(img_dir) if not file.endswith('.DS_Store')] ### --- ignore mac created hidden file


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
    
### ***************************************************************
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
# Complete build
# ---------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
 
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        
        # Residual Blocks
        self.layer1 = self._make_layer(RsBlock, 64, 2, stride=1)  
        self.layer2 = self._make_layer(RsBlock, 128, 2, stride=2) 
        self.layer3 = self._make_layer(RsBlock, 256, 2, stride=2) 
        self.layer4 = self._make_layer(RsBlock, 512, 2, stride=2)
        
        # Pooling and fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(512 * RsBlock.expansion, 6) 

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential( 
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
### *********************************************************************

# Initialize the model
net = Net()

# Load the trained weights
checkpoint = torch.load("saved_best_model_weights/FL_model_best_11_23_2023.pth") ### --- dir of saved model weights biases 
net.load_state_dict(checkpoint)
net.to(device)

# Load the test dataset
path = f"images/test_test"
test_dataset = SatelliteImageDataset(path,True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

########### Dir to save the images
save_dir = "gradcam_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
grad_cam = GradCam(model=net, layer_name="layer4") ######## initialize gradcam
###########

#####----------------------------------------------------------------------

correct = 0
total = 0
y_true = []
y_pred = []


for idx, (images, labels) in enumerate(test_loader):    
    images = images.to(device)
    labels = labels.to(device)
    
    net.eval()  # Set the model to evaluation mode   
    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
    #####----------------------------------------------------------------------
    ###### Get the most probable class if not known
    class_idxs = torch.argmax(outputs, dim=1)
    ###### Generate CAM for this class
    for img_num, class_idx in enumerate(class_idxs):
            
        # Enable gradients for images
        input_image = images[img_num:img_num+1].clone().detach()
        input_image.requires_grad_(True)
        
        # Forward and backward passes    
        net.zero_grad()    
        output = net(input_image)
        one_hot_output = torch.zeros(1, output.size()[-1], device=device)
        one_hot_output[0][class_idx] = 1
            
        output.backward(gradient=one_hot_output)
            
        cam = grad_cam.generate_cam(input_image, class_idx.item())


        # Process the image and CAM for visualization
        img = images[img_num].cpu().numpy().transpose(1, 2, 0)  # images were normalized
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to 0-1
        
        # Apply heatmap to the image
        cam_image = apply_heatmap(img, cam, alpha=0.5)
        
        # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # heatmap = np.float32(heatmap) / 255
        # cam_image = heatmap + np.float32(img)
        # cam_image = cam_image / np.max(cam_image)

        # Convert from RGB to BGR format
        cam_image_bgr = cv2.cvtColor(np.uint8(255 * cam_image), cv2.COLOR_RGB2BGR)

        # Get original image file name and append 'gc' label
        original_file_name = test_dataset.img_labels[idx * test_loader.batch_size + img_num]
        modified_file_name = original_file_name.replace(".tif", "_gc.tif")  # --- save as same name

        # Save the image
        cv2.imwrite(os.path.join(save_dir, modified_file_name), cam_image_bgr)

print(f"Grad-CAM images saved to {save_dir}")
        
# Calculate accuracy
accuracy = 100 * correct / total
print(f'Accuracy on test images: {accuracy:.2f}%')
### ***************************************************************
cm = confusion_matrix(y_true, y_pred)
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
plt.savefig('best_model_SCF_best_11_23_2023t.png')  # Save the standard confusion matrix plot
plt.show()

# Plotting normalized confusion matrix
plt.figure(figsize = (12,7))
sn.heatmap(df_cm2, annot=True, cmap='Blues', fmt='.2%')
plt.title('Normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('best_model_NCF_best_11_23_2023t.png')  # Save the normalized confusion matrix plot
plt.show()

    
