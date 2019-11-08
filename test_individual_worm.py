
import pandas as pd 
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.dataset as dataset
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib as mpl

"""
@filename: test_individual_worm.py
@authors: Dan Griffith, Zach Pincus (zplab)
@project: CNN for deep phenotypic analysis of C. elegans -- DG Rotation project

Usage:
Use this script when constructing a scatterplot of the regressed values of a single worm over 
time. The top section is for using previously saved network weights to calculate a regression
value for each image in the dataset, then saving those predictions into the original csv file.
The bottom section is for constructing the individual worm scatterplots. Comment out the section
that you are not intending to use.


User-specified parameters that may require updating:
--image_dir, csv_dir, weights_dir
--csv_files, weight_paths
--CALC_MEAN, CALC_STD (can calculate for new datasets, but most likely generalizeable for all worms)

"""

##########################################
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

############################################

# User-specified paths
image_dir = '/mnt/lugia_array/Griffith_Dan/worm_images/all_images_mask/'
csv_dir = '/mnt/lugia_array/Griffith_Dan/worm_images/all_images_mask/'
csv_files = ['matt0v_image_data.csv', 'matt1v_image_data.csv', 'matt2v_image_data.csv', 
                        'matt3v_image_data.csv', 'matt4v_image_data.csv']

weights_dir = '/mnt/lugia_array/Griffith_Dan/saved_cnn_states/'
weight_paths = ['matt0_age.pt', 'matt1_age.pt', 'matt2_age.pt', 
                        'matt3_age.pt', 'matt4_age.pt']


# ResNet50 base CNN
model_conv=torchvision.models.resnet50()
# Set fully connected layer architecture
num_ftrs = model_conv.fc.in_features
fc_layers = nn.Sequential(
                nn.Linear(num_ftrs, 1) 
                ) 
model_conv.fc = fc_layers

# Previously calculated mean and std of images' pixel intensities. Used for normalization
CALC_MEAN = [0.490]*3
CALC_STD = [0.209]*3
data_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CALC_MEAN, std=CALC_STD)
    ])

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Iterate through cross-validation csv files, add regressed value to new column in csv file and re-save
for num_file in range(5):
    csv_filename = csv_dir + csv_files[num_file]
    df = pd.read_csv(csv_filename, usecols=[1,2,3,4,5,6,7,8], dtype={'name':'str'})   # Note: may need to adjust usecols list 
    model_conv.load_state_dict(torch.load(weights_dir + weight_paths[num_file]))
    model_conv = model_conv.to(device)
    image_dataset = WormRegressionDataset(image_dir, csv_filename, target='age', transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=4)
    all_outputs = np.array([])
    all_labels = np.array([])

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model_conv.eval()
        outputs = model_conv(inputs)
        labels = labels.float().view(-1, 1)

        all_outputs = np.append(all_outputs, outputs.detach().cpu().numpy())
        all_labels = np.append(all_labels, labels.cpu().numpy())

    # Add new column to dataframe and save dataframe to file
    df['predicted_age'] = all_outputs
    df.to_csv(csv_filename)
  

"""
# Comment out the top section when you want to run the code below:
#########################################
# Create scatterplot of individual worm using regressed values

# Set these to select which worm you want to make a scatterplot for
# Alternatively, could wrap this code in loops to make scatterplots for multiple
num_file = 4
i = 0

csv_filename = csv_dir + csv_files[num_file]
df = pd.read_csv(csv_filename, usecols=[1,2,3,4,5,6,7,8], dtype={'name':'str'})
worms = pd.unique(df['name'])

name = worms[i]
worm_df = df[df['name'] == name]
print(name)
plt.figure(i)
plt.title("True vs predicted hours remaining: Worm " + name, fontsize=16)
plt.xlabel("True hours remaining")
plt.ylabel("Predicted hours remaining")
plt.xlim(left=0, right=800)
plt.ylim(bottom=0, top=800)
plt.plot([0, 800], [0, 800], 'k-')
plt.scatter(worm_df['hours_remaining'].values, worm_df['predicted_hours'].values)
plt.savefig('/mnt/lugia_array/Griffith_Dan/figures/matt0_individualWorms' + name + '_scatter.png')
plt.show()
i += 1
"""
