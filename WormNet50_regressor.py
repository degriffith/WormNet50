# WormNet50_regressor.py

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
@filename: WormNet50_regressor.py
@authors: Dan Griffith, Zach Pincus (zplab)
@project: CNN for deep phenotypic analysis of C. elegans -- DG Rotation project

Usage:
Trains a CNN regressor for a target value on a training set of worm images and validates network
performance on a validation set of images. Unlike the WormNet50 classifier, when regressing 
there are less restrictions on the directory structure that the worm images are located in. However, 
along with the images, the user does need to provide csv files for both the training and validation 
sets that contain the following information:

--"name" column, which specifies which worm a particular image came from (e.g. "002")
--"timepoint" column, specifying the name of the timepoint for that image (e.g. "2017-09-23t0648")
--"<target value>" column, specitying the true regression value for that image (e.g. for 
    "hours_remaining", the value "455.2")

The csv files can also have other columns, but this is not necessary. Each row in the csv file 
corresponds to a particular worm image.

The script will train the network, and plot the training performance over time. This script
will also construct a scatterplot of the predicted vs true values for images in the validation
set.

"""

##################### User specified parameters ########################

# Batch size for training (change depending on how much memory you have)
BATCH_SIZE = 20

# Number of epochs to train for
NUM_EPOCHS = 20

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
FEATURE_EXTRACT = False

# learning rate and momentum parameters
LEARN_RATE = 0.00001
MOMENTUM = 0.9

# Target variable to regress on
TARGET = 'hours_remaining'

# Means and standard dev. vectors to normalize the input images
ORIG_MEAN = [0.485, 0.456, 0.406]
ORIG_STD = [0.229, 0.224, 0.225]
# Calculated mean and standard deviation from the pixel 
# intensity distributions of non-masked worm images
CALC_MEAN = [0.490]*3
CALC_STD = [0.209]*3

# Which worm 'segments' to be cropped during regression: 'head', 'middle', or 'tail'
CROPS = []

# Architecture of final fully connected layers
FC_TYPE = 'fc_type_3'

# Root directory where images are located
DATA_DIR = '/mnt/lugia_array/Griffith_Dan/worm_images/all_images_mask/'

# Training and/or testing the network
DATA_PHASES = ['train', 'val']

"""
Each of the keys in this dictionary correspond to items in DATA_PHASES. Each of the values 
is a keyword to identify the name of the csv file for that phase of training/validation/testing. 
Note that this script assumes that all csv files called have a filename in the format of:
<keyword>_image_data.csv
"""
PHASES_DICT = {'train': 'matt1t', 'val':'matt1v'}

########################## Functions ###############################

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

# Train and validate the network from the data and true regression values
def train_model(model, dataloaders, criterion, optimizer, num_epochs, phases=['train', 'val']):
    since = time.time()

    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')     # Minimize loss

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
                    labels = labels.float().view(-1, 1)

                    loss = criterion(outputs, labels)

                    # Not necessary to calculate accuracy like this, but can be informative
                    # Change the 10.0 to a relevant number for each target value
                    close_guesses = torch.sum(abs(outputs - labels) < 10.0)                   

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics (loss * batch size)        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += close_guesses

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    if 'val' in phases:
        model.load_state_dict(best_model_wts)

    return model, train_loss_history, val_loss_history

"""
Args:
    root_dir (string): Directory with all the images.
    csv_file (string): Path to the csv file containing target values. Requires 'name' and 'timepoint'
                        fields in addition to the target value
    target (string): Name of the target value for indexing the pandas dataframe
    transform (callable, optional): Optional transform to be applied
        on a sample.
"""
class WormRegressionDataset(dataset.Dataset):
    def __init__(self, root_dir, csv_file, target, transform=None):

        self.root_dir = root_dir
        self.target_values_frame = pd.read_csv(csv_file, dtype={'name': 'str'})
        """
        # Use only when excluding extremely long lived worms
        outlier_names = ['08','15','47','64']
        self.target_values_frame = self.target_values_frame[~self.target_values_frame['name'].isin(outlier_names)]
        """
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.target_values_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir + self.target_values_frame['name'].iloc[idx] + '_' + self.target_values_frame['timepoint'].iloc[idx] + '.png'       
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        return image, self.target_values_frame[self.target].iloc[idx]

# Callable pytorch transform class that crops out major segments of the worm image
class SegmentCrop(object):
    '''
    "segments" is a list containing any combination of: 'head', 'middle', or 'tail'
    "segment_divisions" refers to the larger pixel number on the dividing line between segments
    '''
    def __init__(self, segments, segment_divisions):
        self.segments = segments
        self.segment_divisions = segment_divisions

    def __call__(self, tensor):
        return segment_crop(tensor, self.segments, self.segment_divisions)

# Crop out major segments of the worm image
def segment_crop(tensor, segments, segment_divisions):
    if "head" in segments:
        tensor[:, 0:segment_divisions[0], :] = 0
    if "middle" in segments:
        tensor[:, segment_divisions[0]:segment_divisions[1], :] = 0
    if "tail" in segments:
        tensor[:, segment_divisions[1]:, :] = 0
    return tensor

#################################################################################
#################################################################################

# Pretrained ResNet50 base CNN
model_conv=torchvision.models.resnet50(pretrained=True)

# Set fully connected layer architecture
num_ftrs = model_conv.fc.in_features
if FC_TYPE == 'fc_type_1':
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, 1000), 
                  nn.Linear(1000, 1)
                ) 
if FC_TYPE == 'fc_type_2':
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, 512), 
                  nn.Linear(512, 1)
                ) 
if FC_TYPE == 'fc_type_3':
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, 1), 
                ) 
if FC_TYPE == 'fc_type_5':
    fc_layers = nn.Sequential(
                  nn.Linear(num_ftrs, 1000), 
                  nn.ReLU(),
                  nn.Linear(1000, 200),
                  nn.ReLU(),
                  nn.Linear(200, 1)
                ) 
model_conv.fc = fc_layers

# Use if loading previously saved weights:
'''
model_conv.load_state_dict(torch.load('/mnt/lugia_array/Griffith_Dan/saved_cnn_states/WormNet50_regressor_wormSplit_fc3_0001_meanScaled.pt'))
print()
print("Loading saved network weights...")
'''

# Freeze weights that we don't want to change
set_parameter_requires_grad(model_conv, FEATURE_EXTRACT)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    DATA_PHASES[0]: transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # optional crop of a particular segment
        SegmentCrop(segments=CROPS, segment_divisions=[69, 156]),
        # Normalize each image's pixel intensities so that the mean=0 and stdev ~=1
        transforms.Normalize(mean=CALC_MEAN, std=CALC_STD),
    ]),
    DATA_PHASES[1]: transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # optional crop of a particular segment
        SegmentCrop(segments=CROPS, segment_divisions=[69, 156]),       # TODO: change segment_divisions to a variable
        transforms.Normalize(mean=CALC_MEAN, std=CALC_STD)
    ]),
}

print()
print("WormNet50 regression task...")
print("Initializing Datasets and Dataloaders...")
print("Crops =", CROPS)
print("Target value =", TARGET)
for x in DATA_PHASES:
    print("\t", x, '=', PHASES_DICT[x] + "_image_data.csv")


# Create training and validation datasets
image_datasets = {x: WormRegressionDataset(DATA_DIR, csv_file=os.path.join(DATA_DIR, PHASES_DICT[x] + "_image_data.csv"), 
                        target=TARGET, transform=data_transforms[x]) for x in DATA_PHASES}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, 
                        shuffle=True, num_workers=4) for x in DATA_PHASES}

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
if FEATURE_EXTRACT:
    params_to_update = []
    for name,param in model_conv.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

print("Params to learn: ", FC_TYPE)
print("Only updating fully connected layers: ", FEATURE_EXTRACT)

print("Deep learning parameters: ")
print("\tDevice: ", device)
print("\tLearning rate: ", LEARN_RATE)
print("\tMomentum: ", MOMENTUM)
print()

# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(params_to_update, lr=LEARN_RATE, momentum=MOMENTUM)

# Setup the loss fxn
criterion = nn.L1Loss()

# Train and evaluate (optionally save the weights)
model_conv, train_hist, val_hist = train_model(model_conv, dataloaders_dict, criterion, optimizer, NUM_EPOCHS, phases=DATA_PHASES)
#torch.save(model_conv.state_dict(), '/mnt/lugia_array/Griffith_Dan/saved_cnn_states/matt4_excludeOutliers_percentRem.pt')

#########################################################################################

# Plot the training curves of training and validation accuracy vs. number
#  vs. number of training epochs for the transfer learning method 
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

plt.figure(0)
plt.plot(range(1, NUM_EPOCHS + 1), train_hist, label="Training")
plt.plot(range(1, NUM_EPOCHS + 1), val_hist, label="Validation")
plt.title("Training and Validation Loss vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(1, NUM_EPOCHS+1, 2.0))
plt.legend()
plt.savefig('/mnt/lugia_array/Griffith_Dan/figures/matt1_learningRate00001.png')


#########
# Scatterplot of predictions vs actual labels

all_outputs = np.array([])
all_labels = np.array([])
for inputs, labels in dataloaders_dict['val']:
    inputs = inputs.to(device)
    labels = labels.to(device)
    model_conv.eval()
    outputs = model_conv(inputs)
    labels = labels.float().view(-1, 1)

    all_outputs = np.append(all_outputs, outputs.detach().cpu().numpy())
    all_labels = np.append(all_labels, labels.cpu().numpy())

# R^2 and p-value of regression estimate
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(all_labels, all_outputs)
print()
print("R^2:", r_value**2)
print("p-value:", "{:.2E}".format(p_value))
print("slope:", slope, "intercept:", intercept)
print()

textstr = '\n'.join((
    r'R^2 = %.3f' % (r_value**2, ),
    r'm = %.2f' % (slope, ),
    r'b = %.2f' % (intercept, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


plt.figure(1)
plt.scatter(all_labels, all_outputs, s=(mpl.rcParams['lines.markersize'] ** 2)/2)

# Comment out 2/3 based on which target value is being regressed on
plt.title("True vs predicted hours remaining", fontsize=16)
plt.xlabel("True hours remaining")
plt.ylabel("Predicted hours remaining")
plt.xlim(left=0, right=800)
plt.ylim(bottom=0, top=800)
plt.text(25, 650, textstr, fontsize=12, bbox=props)
plt.plot([0, 800], [intercept, 800*slope + intercept], 'k--')
plt.plot([0, 800], [0, 800], 'k-')
'''
plt.title("True vs predicted percent remaining", fontsize=16)
plt.xlabel("True percent remaining")
plt.ylabel("Predicted percent remaining")
plt.text(0, 80, textstr, fontsize=12, bbox=props)
plt.plot([0, 100], [intercept, 100*slope + intercept], 'k--')
plt.plot([0, 100], [0, 100], 'k-')

plt.title("True vs predicted age", fontsize=16)
plt.xlabel("True age")
plt.ylabel("Predicted age")
plt.text(25, 650, textstr, fontsize=12, bbox=props)
plt.xlim(left=0, right=800)
plt.ylim(bottom=0, top=800)
plt.plot([0, 800], [intercept, 800*slope + intercept], 'k--')
plt.plot([0, 800], [0, 800], 'k-')
'''

plt.savefig('/mnt/lugia_array/Griffith_Dan/figures/matt0_middleTailCrop_age_scatter.png')

