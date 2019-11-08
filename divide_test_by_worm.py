# divide_test_by_worm.py

import pandas as pd
import numpy as np
import os

"""
@filename: divide_test_by_worm.py
@authors: Dan Griffith, Zach Pincus (zplab)
@project: CNN for deep phenotypic analysis of C. elegans -- DG Rotation project

Usage:
This script is not meant to be run from the terminal. Instead it is a collection of potentially helpful
sub-scripts for dividing a dataset into training/test sub-sets on a worm-by-worm basis.

User-specified parameters:
--csv_file
--image_dir

"""

##### Divide a dataset into only a training and validation set (not split for cross-validation)

csv_file = '/mnt/lugia_array/Griffith_Dan/worm_images/all_images_mask/image_data_all_masked.csv'
df = pd.read_csv(csv_file, usecols=[1,2,3,4,5], dtype={'name':'str'}) # May need to adjust usecols

worms = df['name'].unique()

val_worms = []
for worm in worms:
    if np.random.rand() < 0.15:
        val_worms.append(worm)

image_dir = '/mnt/lugia_array/Griffith_Dan/worm_images/data_split_by_individual/'
train_dir = '/mnt/lugia_array/Griffith_Dan/worm_images/data_split_by_individual/train/'
val_dir = '/mnt/lugia_array/Griffith_Dan/worm_images/data_split_by_individual/val/'

df_train = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])
df_val = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])

for i in range(len(df)):
    name = df.iloc[i]['name']
    timepoint = df.iloc[i]['timepoint']
    filename = name + '_' + timepoint + '.png'

    if os.path.exists(image_dir + filename):
        if name in val_worms:
			# Add to val directory, val df
            os.rename(image_dir + filename, val_dir + filename)
            df_val = df_val.append(df.iloc[i])
        else:
			# Add to train directory, train df
            os.rename(image_dir + filename, train_dir + filename)
            df_train = df_train.append(df.iloc[i])

df_train.to_csv(image_dir + 'train_image_data.csv')
df_val.to_csv(image_dir + 'val_image_data.csv')

#########################################
# Split a dataset into 5 ~even subsets on a worm-by-worm basis (e.g. if 50 worms total, each set should get 9-11)
# For cross-validation, each training set will have 4 of these subsets, and the one left out is the validation set

import pandas as pd
import numpy as np

image_dir = '/mnt/lugia_array/Griffith_Dan/worm_images/all_images_mask/'
csv_file = '/mnt/lugia_array/Griffith_Dan/worm_images/all_images_mask/matt_image_data.csv'
df = pd.read_csv(csv_file, usecols=[1,2,3,4,5], dtype={'name':'str'}) # May need to adjust usecols

# Repeat this block until each group has approximately the same number of worms
a = pd.unique(df['name'])
d = {}
b = np.random.randint(0,5,len(a))
print(58 - np.count_nonzero(b))
print(58 - np.count_nonzero(b-1))
print(58 - np.count_nonzero(b-2))
print(58 - np.count_nonzero(b-3))
print(58 - np.count_nonzero(b-4))



# Create the new csv files for the train/test sets for each of these 5 groups
for i in range(len(a)):
    d[a[i]] = b[i]

df['age'] = df['lifespan'] - df['hours_remaining']

matt0t = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])
matt0v = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])

matt1t = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])
matt1v = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])

matt2t = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])
matt2v = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])

matt3t = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])
matt3v = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])

matt4t = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])
matt4v = pd.DataFrame(columns=['name', 'timepoint', 'lifespan', 'hours_remaining', 'percentage_remaining', 'age'])

for j in range(len(df)):
    name = df.iloc[j]['name']
    idx = d[name]
    if idx == 0:
        matt0v = matt0v.append(df.iloc[j])
    else:
        matt0t = matt0t.append(df.iloc[j])

for j in range(len(df)):
    name = df.iloc[j]['name']
    idx = d[name]
    if idx == 1:
        matt1v = matt1v.append(df.iloc[j])
    else:
        matt1t = matt1t.append(df.iloc[j])

for j in range(len(df)):
    name = df.iloc[j]['name']
    idx = d[name]
    if idx == 2:
        matt2v = matt2v.append(df.iloc[j])
    else:
        matt2t = matt2t.append(df.iloc[j])

for j in range(len(df)):
    name = df.iloc[j]['name']
    idx = d[name]
    if idx == 3:
        matt3v = matt3v.append(df.iloc[j])
    else:
        matt3t = matt3t.append(df.iloc[j])

for j in range(len(df)):
    name = df.iloc[j]['name']
    idx = d[name]
    if idx == 4:
        matt4v = matt4v.append(df.iloc[j])
    else:
        matt4t = matt4t.append(df.iloc[j])
     
matt0t.to_csv(image_dir + 'matt0t_image_data.csv')
matt0v.to_csv(image_dir + 'matt0v_image_data.csv')

matt1t.to_csv(image_dir + 'matt1t_image_data.csv')
matt1v.to_csv(image_dir + 'matt1v_image_data.csv')

matt2t.to_csv(image_dir + 'matt2t_image_data.csv')
matt2v.to_csv(image_dir + 'matt2v_image_data.csv')

matt3t.to_csv(image_dir + 'matt3t_image_data.csv')
matt3v.to_csv(image_dir + 'matt3v_image_data.csv')

matt4t.to_csv(image_dir + 'matt4t_image_data.csv')
matt4v.to_csv(image_dir + 'matt4v_image_data.csv')

