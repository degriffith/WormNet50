# WormNet50_classifier.py

import torch
import torch.nn as nn
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.dataset as dataset
import copy
import os
import time
import pandas as pd


"""
@filename: WormNet50_classifier.py
@authors: Dan Griffith, Zach Pincus (zplab)
@project: CNN for deep phenotypic analysis of C. elegans -- DG Rotation project

Usage:
Trains a CNN classifier and validates network performance on pre-divided worm images based on
their specified class. The images should be set up in the following directory structure
(2 class example):

/dir_root
----/train
--------/class0
------------[All class0 training images]
--------/class1
------------[All class1 training images]
----/val
--------/class0
------------[All class0 validation images]
--------/class1
------------[All class1 validation images]

The script will train the network, and plot the training performance over time. This script
will also construct a confusion matrix of the predicted vs true classes for images in the validation
set.

"""

##################### User specified parameters ########################

# Batch size for training (change depending on how much memory you have)
BATCH_SIZE = 20

# Number of epochs to train for
NUM_EPOCHS = 10

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
FEATURE_EXTRACT = False

# Classification problem when False, regression problem when True
NUM_CLASSES = 5   

# learning rate and momentum parameters
LEARN_RATE = 0.001
MOMENTUM = 0.9

# Means and standard dev. vectors to normalize the input images
ORIG_MEAN = [0.485, 0.456, 0.406]
ORIG_STD = [0.229, 0.224, 0.225]
CALC_MEAN = [0.490]*3
CALC_STD = [0.209]*3

# Architecture of final fully connected layers
FC_TYPE = 'fc_type_1'

# Root directory where images are located
DATA_DIR = '/mnt/lugia_array/Griffith_Dan/worm_images/five_classes/'

# Training and/or testing the network
DATA_PHASES = ['train', 'val']

######################### Functions ##############################

# Set which layers to update:
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        ## Freezing all but the new fully connected layers
        ct = 0
        for name, child in model_conv.named_children():
            ct += 1
            if ct < 10:
                for name2, params in child.named_parameters():
                    params.requires_grad = False

"""
Train and validate the model using images, labels, and specified machine learning parameters.
Returns the weights from the best performing network, as well as the history of training / 
validation loss per epoch.
"""
def train_model(model, dataloaders, criterion, optimizer, num_epochs, phases=['train', 'val']):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # Maximize accuracy and/or minimize loss

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                        
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics (loss * batch size)        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    if 'val' in phases:
        model.load_state_dict(best_model_wts)

    print('Best val Acc: {:4f}'.format(best_acc))
    return model, train_acc_history, val_acc_history
  

#################################################################################
#################################################################################

# Pretrained ResNet50 base CNN
model_conv=torchvision.models.resnet50(pretrained=True)

# Set fully connected layer architecture
num_ftrs = model_conv.fc.in_features
# Different possible fc architectures to explore
if FC_TYPE == 'fc_type_1':
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, 1000), 
                  nn.Linear(1000, NUM_CLASSES)
                ) 
if FC_TYPE == 'fc_type_2':
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, 512), 
                  nn.Linear(512, NUM_CLASSES)
                ) 
if FC_TYPE == 'fc_type_3':
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, NUM_CLASSES), 
                ) 
if FC_TYPE == 'fc_type_4':
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, 1000), 
                  nn.LeakyReLU(),
                  nn.Linear(1000, 200),
                  nn.LeakyReLU(),
                  nn.Linear(200, NUM_CLASSES)
                ) 
if FC_TYPE == 'fc_type_5':
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, 1000), 
                  nn.ReLU(),
                  nn.Linear(1000, 200),
                  nn.ReLU(),
                  nn.Linear(200, NUM_CLASSES)
                ) 
if FC_TYPE == 'fc_type_6':
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, 1000), 
                  nn.ReLU(),
                  nn.Linear(1000, 20),
                  nn.ReLU(),
                  nn.Linear(20, NUM_CLASSES)
                ) 
if FC_TYPE == 'fc_type_7':      
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, 1000), 
                  nn.Linear(1000, 2),
                  nn.Linear(2, 2),
                  nn.Linear(2, NUM_CLASSES)
                ) 
# Replace the fully connected layer of the model with our constructed fully connected layers
model_conv.fc = fc_layers

# Freeze weights that we don't want to change
set_parameter_requires_grad(model_conv, FEATURE_EXTRACT)

# Data cropping and normalization for training and validation
# Optionally could augment images for training
data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CALC_MEAN, std=CALC_STD),
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CALC_MEAN, std=CALC_STD)
    ]),
}

print()
print("Classification problem...")
print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in DATA_PHASES}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in DATA_PHASES}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_conv = model_conv.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_conv.parameters()
print("Params to learn: ", FC_TYPE)
if FEATURE_EXTRACT:
    params_to_update = []
    for name,param in model_conv.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_conv.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

print("Deep learning parameters: ")
print('Device: ', device)
print("\tLearning rate: ", LEARN_RATE)
print("\tMomentum: ", MOMENTUM)

# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(params_to_update, lr=LEARN_RATE, momentum=MOMENTUM)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_conv, train_hist, val_hist = train_model(model_conv, dataloaders_dict, criterion, optimizer, NUM_EPOCHS, phases=DATA_PHASES)
# Save network weights
#torch.save(model_conv.state_dict(), '/mnt/lugia_array/Griffith_Dan/saved_cnn_states/fiveclass_10_fc1_001.pt')

######################### Make plots ################################

# Plot the training curves of training and validation accuracy vs. number
#  vs. number of training epochs for the transfer learning method 
from matplotlib import pyplot as plt
import numpy as np

plt.figure(0)
plt.plot(range(1, NUM_EPOCHS + 1), train_hist, label="Training")
plt.plot(range(1, NUM_EPOCHS + 1), val_hist, label="Validation")
plt.title("Training and Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, NUM_EPOCHS+1, 1.0))
plt.legend()
plt.savefig('/mnt/lugia_array/Griffith_Dan/figures/fiveclassAccuracy_10_fc1_001.png')

##############

# Confusion matrix of predictions vs true labels
from sklearn.metrics import confusion_matrix

all_preds = np.array([])
all_labels = np.array([])
bad_pred_images = np.random.rand(1,224,224)

for inputs, labels in dataloaders_dict['val']:
    inputs = inputs.to(device)
    labels = labels.to(device)

    model_conv.eval()
    outputs = model_conv(inputs)
    _, preds = torch.max(outputs, 1)

    all_preds = np.append(all_preds, preds.detach().cpu().numpy())
    all_labels = np.append(all_labels, labels.detach().cpu().numpy())

print()

cm = confusion_matrix(all_labels, all_preds)
print(cm)
classes = ['0-90 h.r.', '90-142 h.r.', '142-190 h.r.', '190-262 h.r.', '262-735 h.r.']

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=classes,
       title='Confusion matrix',
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

fig.savefig('/mnt/lugia_array/Griffith_Dan/figures/fiveclassCM_10_fc1_001.png')
