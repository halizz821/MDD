
#%% Import Libraries"""
# PyTorch version: 2.2.0
# CUDA available: True
# CUDA version (compiled): 12.1

import numpy as np

from scipy.io import loadmat
 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset        
    
from torchinfo import summary
from time import time
import os

#%% GLOBAL VARIABLES
windowSize = 9

#%% Functions
# 
from MultiscaleDeformableDenseNet import ExtractPatches, ScaleData, train_split_n_sample_perclass


#%% Load Data

dir_path = 'G:\\DeepLearning-HSI classification\\My ViT+Conv\\CamPha\\img\\'


# # Open a multi-band TIFF image Load the MATLAB file
# Get the directory where main.py is located

base_dir = os.getcwd()

# Construct full path to the .mat file
data_address = os.path.join(base_dir, 'CamPha', 'CamPha_image.mat')
mat = loadmat(data_address)
image_array = mat['CamPha_image']
data = image_array.astype(np.float16)

from einops import rearrange

# Change the order of dimensions using einops
# Here, we're swapping the first and second dimensions
data = rearrange(data, 'i j k -> k i j')


# # Open a  ground truth (gt)
data_address = os.path.join(base_dir, 'CamPha', 'CamPha_label.mat')
mat = loadmat(data_address)
gt = mat['CamPha_label']
print(f'Data Shape: {data.shape[1:3]}\nNumber of Bands: {data.shape[0]}')




#%%"""# Pre-processing

# ################# Extract patches of the the image
X, labels, pos_in_image = ExtractPatches(data, GT=gt, windowSize=windowSize)

################# Encode labels of each class    
X_train, y_train = train_split_n_sample_perclass(X, labels, per_class=200,
                                                  randomstate=42, # change it for a different train/val configuration 
                                                  nb_class= np.max(gt))

enc = OneHotEncoder()
y_train=enc.fit_transform(y_train.reshape(-1, 1)).toarray() #turn labels to categorical (i.e., each label is represented by a vector) usig  OneHotEncoder method

################# Split samples into: 1) Training, 2) Validation 
   
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    stratify=y_train,
                                                    train_size=0.8,
                                                    random_state=42 # change it for a different train/val configuration 
                                                    )    

# ################ Scaling

# Find the minimum and maximum values of the array
min_value = np.min(X_train)
max_value = np.max(X_train)

# Scale the array between 0 and 1
X_train=ScaleData(X_train,min_value,max_value)
X_val=ScaleData(X_val,min_value,max_value)


print(f"X_train: {X_train.shape}\ny_train: {y_train.shape}\nX_validation: {X_val.shape}\ny_validation: {y_val.shape}\n")

        
#%% Train Model"""

batch_size=256

################################################
# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert NumPy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float).to(device)

# Create DataLoader for training, validation, and test sets
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#%% Initialize the model and loss function

from MultiscaleDeformableDenseNet import MDD

input_shape = X_train[0].shape                       # input shape of  model  e.g.(200,21,21)
nb_classes= y_train.shape[1]                         # output shape of model

model = MDD(nb_input_channels=input_shape[0], nb_blocks=2, nb_layers_in_block=[4,4], growth_rate=12, 
            nb_classes = nb_classes, botneck_scale=4,                     
            dropout_rate= None, transition_factor= 0.5) 


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr= 0.0001 , weight_decay=0.0005)


input = (1,) + input_shape
print(summary(model, input))
   
model = model.to(device) 

#%% Train the model

train_loss_values = []
val_loss_values = []

nb_epoch=100
best_val_loss = float('inf')


for epoch in range(nb_epoch):
    start_time = time()
    
    model.train()
    train_loss = 0

    for inputs, labels in train_loader :  #tqdm(train_loader)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.max(1)[1]).sum().item()

    val_loss /= len(val_loader)
    accuracy = correct / total * 100
    
    epoch_time= time()-start_time 
    print(f'Epoch [{epoch + 1}/{nb_epoch}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}, Time: {epoch_time:.2f}')

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_2.pth')


    # Append the training and validation loss values for plotting
    train_loss_values.append(train_loss / len(train_loader))
    val_loss_values.append(val_loss)


# test model performace
from MultiscaleDeformableDenseNet import evaluate_model_accuracy
metrics = evaluate_model_accuracy(
    model=model,
    data=data,
    gt=gt,
    windowSize=windowSize,
    enc=enc,
    ScaleData=ScaleData,
    min_value=min_value,
    max_value=max_value,
    batch_size=batch_size,
    chunk_size=2000,# change it based on you GPU memory
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    model_path='best_model_2.pth',
   
)