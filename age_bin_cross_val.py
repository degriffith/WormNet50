# age_bin_cross_val.py

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
@filename: age_bin_cross_val.py
@authors: Dan Griffith, Zach Pincus (zplab)
@project: CNN for deep phenotypic analysis of C. elegans -- DG Rotation project

Usage:
Take the results from cross-validation and combine them into a larger scatterplot. Additionally, can 
split a previously-calculated regression over the dataset into individual age bins. Can recompute
R^2 on each of these age bins and create a scatterplot for each age bin.

"""

################# User-specified parameters #######################

image_dir = '/mnt/lugia_array/Griffith_Dan/worm_images/all_images_mask'
csv_dir = '/mnt/lugia_array/Griffith_Dan/worm_images/all_images_mask/'
csv_files = ['matt0v_image_data.csv', 'matt1v_image_data.csv', 'matt2v_image_data.csv', 
                      	'matt3v_image_data.csv', 'matt4v_image_data.csv']

weights_dir = '/mnt/lugia_array/Griffith_Dan/saved_cnn_states/'
weight_paths = ['matt0_hoursRem.pt', 'matt1_hoursRem.pt', 'matt2_hoursRem.pt', 
						'matt3_hoursRem.pt', 'matt4_hoursRem.pt']

TARGET = "hours_remaining"
CROPS = []

################## Functions #########################

class WormRegressionDataset(dataset.Dataset):

    def __init__(self, root_dir, csv_file, files_list, target, transform=None):
        self.root_dir = root_dir
        self.target_values_frame = pd.read_csv(csv_file, dtype={'name': 'str'})
        self.files_list = files_list
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir + '/' + self.files_list[idx]

        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        target_val = self.target_values_frame[self.target_values_frame['name'] + '_' +
        	self.target_values_frame['timepoint'] + '.png' == self.files_list[idx]][self.target].values[0]

        return image, target_val

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

def segment_crop(tensor, segments, segment_divisions):
    if "head" in segments:
        tensor[:, 0:segment_divisions[0], :] = 0
    if "middle" in segments:
        tensor[:, segment_divisions[0]:segment_divisions[1], :] = 0
    if "tail" in segments:
        tensor[:, segment_divisions[1]:, :] = 0
    return tensor

################################################################

# Each value in the dictionary corresponds to an age bin
scatterplot_points = {}
for i in range(16):
	scatterplot_points[i] = np.array([[0],[0]])

#### Load pretrained network to evaluate worm images from within a certain age bin on
# ResNet50 base CNN
model_conv=torchvision.models.resnet50()
# Set fully connected layer architecture
num_ftrs = model_conv.fc.in_features
fc_layers = nn.Sequential(
                nn.Linear(num_ftrs, 1) 
                ) 
model_conv.fc = fc_layers

# Tranformations to run on the images prior to inputting into the network
CALC_MEAN = [0.490]*3
CALC_STD = [0.209]*3
data_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        SegmentCrop(segments=CROPS, segment_divisions=[69, 156]),
        transforms.Normalize(mean=CALC_MEAN, std=CALC_STD)
    ])

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

####################################################################

# Iterate through the csv files
for num_file in range(len(csv_files)):
	csv_filename = csv_dir + csv_files[num_file]

	df = pd.read_csv(csv_filename, dtype={'name':'str'})
	#age = df['lifespan'] - df['hours_remaining']  # If 'age' is not already present in csv file
	#df['age'] = age
	df = df.sort_values(by='age')

	# Bins of ages 3-18 days old
	age_cutoffs = np.linspace(3*24, 20*24, 18)

	# Find out which indices separate each of the agebins
	cutoff_indices = []
	j = 0
	for i in range(len(df)):
		if df.iloc[i]['age'] >= age_cutoffs[j] and df.iloc[i]['age'] < age_cutoffs[j+1]:
			cutoff_indices.append(i)
			j+=1
		if j == len(age_cutoffs) - 1:
			break

	# Iterate through age bins, selecting the appropriate worm image files
	images = []
	for j in range(len(cutoff_indices) - 1):
		files = []
		# Add image filenames to list based on age bin
		for k in range(cutoff_indices[j], cutoff_indices[j+1]):
			name = df.iloc[k]['name']
			timepoint = df.iloc[k]['timepoint']
			file = name + '_' + timepoint + '.png'
			files.append(file)

		images.append(files) 

	# Load trained network weights
	model_conv.load_state_dict(torch.load(weights_dir + weight_paths[num_file]))
	print()
	print("Loading saved network weights...")

	# Send the model to GPU
	model_conv = model_conv.to(device)

	#### Loop over each age bin
	for i in range(len(images)):
		
		# Make dataloader for that particular age bin
		image_dataset = WormRegressionDataset(image_dir, csv_filename, images[i],
						target=TARGET, transform=data_transforms)
		dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=20, shuffle=True, num_workers=4)

		#### Evaluate network regression task on dataloader
		# Scatterplot of predictions vs actual labels
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

		scatterplot_points[i] = np.hstack((scatterplot_points[i], np.vstack((all_labels, all_outputs))))

all_points = np.array([[0],[0]])

########################## Plotting ###################################
# *** NOTE: Cannot plot both figures at same time, so one needs to be commented out at a time to prevent an error

# Plot 1: create scatterplots for each of the age bins:

# Share both X and Y axes with all subplots
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle("True vs predicted hours remaining", fontsize=45)
fig.text(0.5, 0.04, 'True hours remaining', ha='center', fontsize=30)
fig.text(0.04, 0.5, 'Predicted hours remaining', va='center', rotation='vertical', fontsize=30)

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

for i in range(16):
	all_points = np.hstack((all_points, scatterplot_points[i][:, 1:]))

	sample_size = len(scatterplot_points[i][0] - 1)
	slope, intercept, r_value, p_value, std_err = stats.linregress(scatterplot_points[i][0, 1:], scatterplot_points[i][1, 1:])

	print("Age bin: " + str(i+2) + " Days old")
	print("Sample size:", sample_size)
	print("R^2:", "{:.3f}".format(r_value**2))
	print("p-value:", "{:.2E}".format(p_value))
	print("slope:", "{:.2f}".format(slope), "intercept:", "{:.2f}".format(intercept))
	print()

	#### Add to plot as a subplot
	textstr = '\n'.join((
        r'R^2 = %.3f' % (r_value**2, ),
        r'n = %d' % (sample_size, )))
	m = i // 4
	n = i % 4


	axs[m, n].scatter(scatterplot_points[i][0, 1:], scatterplot_points[i][1, 1:], s=(mpl.rcParams['lines.markersize'] ** 2)/2)

	if p_value < 0.05:
		subtitle_text = " Days old **"
	else:
		subtitle_text = " Days old"

	
	axs[m, n].text(50, 650, textstr, fontsize=10, bbox=props)
	axs[m, n].set_title(str(i+3) + subtitle_text)
	axs[m, n].set_xlim([0,800])
	axs[m, n].set_ylim([0,800])
	axs[m, n].plot([0, 800], [intercept, 800*slope + intercept], 'k--')
	axs[m, n].plot([0, 800], [0, 800], 'k-')
	'''
	axs[m, n].text(10, 80, textstr, fontsize=10, bbox=props)
	axs[m, n].set_title(str(i+3) + subtitle_text)
	axs[m, n].set_xlim([0,100])
	axs[m, n].set_ylim([0,100])
	axs[m, n].plot([0, 100], [intercept, 100*slope + intercept], 'k--')
	axs[m, n].plot([0, 100], [0, 100], 'k-')
	
	axs[m, n].set_title(str(i+2) + subtitle_text)
	axs[m, n].text(50, 650, textstr, fontsize=10, bbox=props)
	axs[m, n].set_xlim([0,800])
	axs[m, n].set_ylim([0,800])
	axs[m, n].plot([0, 800], [intercept, 800*slope + intercept], 'k--')
	axs[m, n].plot([0, 800], [0, 800], 'k-')
	'''
fig.tight_layout(rect=[0.1, 0.1, 1, 0.9])
plt.savefig('/mnt/lugia_array/Griffith_Dan/figures/agebins_18days_excludeOutliers_hoursRem.png')

########
# Plot 2: create a scatterplot combining regressions from each of the cross-validations

all_points = all_points[:, 1:]
plt.scatter(all_points[0]/24, all_points[1]/24, s=(mpl.rcParams['lines.markersize'] ** 2)/2)

slope, intercept, r_value, p_value, std_err = stats.linregress(all_points[0]/24, all_points[1]/24)
print()
print("All images:")
print("R^2:", r_value**2)
print("p-value:", "{:.2E}".format(p_value))
print("slope:", slope, "intercept:", intercept)
print()

textstr = '\n'.join((
    r'R^2 = %.3f' % (r_value**2, ),
    r'm = %.2f' % (slope, ),
    r'b = %.2f' % (intercept, )))
'''
plt.title("True vs predicted hours remaining", fontsize=16)
plt.xlabel("True hours remaining")
plt.ylabel("Predicted hours remaining")
plt.xlim(left=0, right=800)
plt.ylim(bottom=0, top=800)
plt.text(25, 650, textstr, fontsize=12, bbox=props)
plt.plot([0, 800], [intercept, 800*slope + intercept], 'k--')
plt.plot([0, 800], [0, 800], 'k-')

plt.title("True vs predicted percent remaining", fontsize=16)
plt.xlabel("True percent remaining")
plt.ylabel("Predicted percent remaining")
plt.text(0, 80, textstr, fontsize=12, bbox=props)
plt.plot([0, 100], [intercept, 100*slope + intercept], 'k--')
plt.plot([0, 100], [0, 100], 'k-')
'''
plt.title("True vs predicted age", fontsize=16)
plt.xlabel("True age")
plt.ylabel("Predicted age")
plt.text(25 / 24, 650 / 24, textstr, fontsize=12, bbox=props)
plt.xlim(left=0, right=800 / 24)
plt.ylim(bottom=0, top=800 / 24)
plt.plot([0, 800 / 24], [intercept / 24, 800*slope / 24 + intercept / 24], 'k--')
plt.plot([0, 800 / 24], [0, 800 / 24], 'k-')


#plt.savefig('/mnt/lugia_array/Griffith_Dan/figures/matt_excludeOutliers_age.png')
