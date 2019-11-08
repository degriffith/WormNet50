# aggregate_and_preprocess_images.py

from elegant import worm_data
from elegant import load_data
from elegant import process_images
from elegant import worm_spline
from ris_widget import ris_widget
from zplib import curve
import pickle
import freeimage 
import numpy as np
import csv

"""
@filename: aggregate_and_preprocess_images.py
@authors: Dan Griffith, Zach Pincus (zplab)
@project: CNN for deep phenotypic analysis of C. elegans -- DG Rotation project

Usage:
This script is designed to gather all scope images from a specified directory and process them so
that they are ready to be inputted into a convolutional neural network. Additionally, this script will
create a csv file that contains pertinent image specific worm information.

User-specified paramenters:
--experimental_root
--worm_annotation_files
--OUTPUT_DIRECTORY

*Note* Adjust or add to the processing pipeline as needed

"""

#################### functions ##########################

##### Adjust the image modal values and scale to [0, 255] #####
def scale_image(image, optocoupler=1, new_mode=24000, new_max=26000, new_min=600, gamma=0.72):
    # step 1: make an image that has a modal value of 24000, yielding a known intensity distribution
    noise_floor = 100
    image_mode = process_images.get_image_mode(image, optocoupler=optocoupler)
    image = image.astype(np.float32)
    image -= noise_floor
    image *= (new_mode - noise_floor) / (image_mode - noise_floor)
    image += noise_floor
   
    # step 2: scale that known distribution down to [0,1] using empirical parameters
    image.clip(min=new_min, max=new_max, out=image)
    image -= new_min
    image /= new_max - new_min
    # now image is in the range [0, 1]
    image **= gamma # optional nonlinear gamma transform

    # step 3: scale to 8bit
    image *= 255
    image = np.round(image)
    image = image.astype(np.uint8)

    return image

### Worm resizing information and functions ###
def to_tck(widths):
       x = np.linspace(0, 1, len(widths))
       smoothing = 0.0625 * len(widths)
       return curve.interpolate.fit_nonparametric_spline(x, widths, smoothing=smoothing)

# Adjust worm frame widths and lengths to standard values
WIDTH_TRENDS = pickle.load(open('/Users/zplab/miniconda/lib/python3.7/site-packages/elegant/width_data/width_trends.pickle', 'rb'))
# Widths of average worm at the age of 5 days (interpolated)
AVG_WIDTHS = np.array([np.interp(5, WIDTH_TRENDS['ages'], wt) for wt in WIDTH_TRENDS['width_trends']])
AVG_WIDTHS_TCK = to_tck(AVG_WIDTHS)

# Resize worm image so that all images in dataset are a uniform size. Set non-masked pixels to zero if specified
def standardized_worm_frame_image(lab_frame_image, center_tck, width_tck, pixel_height, 
        pixel_width, useMask=False, average_widths_tck=AVG_WIDTHS_TCK):
    WORM_WIDTH= 64
    WORM_PAD = 38

    worm_width_factor = (pixel_height - WORM_PAD) / WORM_WIDTH
    new_avg_width_tck = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1] * worm_width_factor, AVG_WIDTHS_TCK[2])
    worm_frame_image = worm_spline.to_worm_frame(lab_frame_image, center_tck, width_tck, 
                                standard_length=pixel_width, standard_width=new_avg_width_tck)
    
    if useMask:
        mask = worm_spline.worm_frame_mask(new_avg_width_tck, worm_frame_image.shape)
        worm_frame_image[mask < 10] = 0
        return worm_frame_image

    else:
        return worm_frame_image

# "Crudely" square a worm-frame image by spliting the worm in thirds and stacking those segments vertically
def square_worm_frame(worm_frame_image):
    # x-axis and y-axis are swapped from numpy matrix -> RisWidget
    pixel_width = len(worm_frame_image)
    pixel_height = len(worm_frame_image[0])

    if not pixel_width % 3 == 0:
        print("Worm length in pixels must be divisible by 3.")
    else:
        width_third = int(pixel_width / 3)

        reshaped_image = np.concatenate((worm_frame_image[0:width_third, :], 
            worm_frame_image[width_third:2*width_third, :], worm_frame_image[2*width_third:, :]), axis=1)

        return reshaped_image

# Stack three grayscale images in a third dimension so that it mimics a RGB image
def create_RGB_stack(gray_image):
    return np.stack([gray_image, gray_image, gray_image], axis=2)


#########################################################################################

# Root containing raw image folders and annotated poses
#experimental_root =  '/Volumes/9karray/Mosley_Matt/20190408_lin-4_spe-9_20C_pos-1'     # Use for Matt's dataset
experimental_root = '/Volumes/lugia_array/20170919_lin-04_GFP_spe-9'                    # Use for Nicolette's dataset 

# Use for Matt's dataset, comment out for Nicolette's
#worm_annotation_files = '/Volumes/9karray/Mosley_Matt/20190408_lin-4_spe-9_20C_pos-1/derived_data/measurements/core_measures/*.tsv'
# Use for Nicolette's, comment out when using Matt's
worm_annotation_files = '/Volumes/lugia_array/20170919_lin-04_GFP_spe-9/measured_health/*.tsv'

### Information on age, lifespan, etc (assumes 'exclude'-flagged worms are already excluded from directory)
worm_annotations = worm_data.read_worms(worm_annotation_files, name_prefix='')
worm_annotations.sort('name') # Optional sort, but easier to follow for debugging

# Use timepoint filter only for Nicolette's dataset
####
worm_tp_dict = {}
timepoints = worm_annotations.get_time_range('timepoint', min_age=24*2, max_age=24*7)
for i, worm in enumerate(worm_annotations):
    tp = timepoints[i]
    worm_tp_dict[worm.name] = tp.T[1]

def timepoint_filter(position_name, timepoint_name, tp_dict = worm_tp_dict):
    if position_name not in tp_dict.keys():
        return False
    if timepoint_name in tp_dict[position_name]:
        return True
    else:
        return False
files = load_data.scan_experiment_dir(experimental_root, timepoint_filter=timepoint_filter)
####
# Use this for Matt's dataset
#files = load_data.scan_experiment_dir(experimental_root)   

# Image annotation data
positions = load_data.read_annotations(experimental_root)
positions = load_data.filter_annotations(positions, load_data.filter_excluded)

# Location of output files
OUTPUT_DIRECTORY = '/Users/zplab/Desktop/DanScripts/worm_images/---'

# Array with pertinent worm information
worm_image_data = [['name', 'timepoint', 'lifespan', 'hours_remaining']]
# Iterate through worms
for worm in worm_annotations:
    worm_name = worm.name
    print(worm_name)

    try:
    	# Check that this worm is also present in positions
    	positions[worm_name]
    except:
    	continue

    worm_lifespan = worm.lifespan
    # Iterate through timepoints for that worm
    for i in range(len(worm.td.timepoint)):

        # Ensure that worm is an adult (adjust filter depending on annotations and dataset)
        if worm.td.age[i] <= 168 and worm.td.age[i] >= 48:           # Use for Nicolette's dataset
        #if int(worm_name) < 67 and worm.td.stage[i] == 'adult':     # Use for Matt's dataset

            image_timepoint = worm.td.timepoint[i]
            hours_remaining = -worm.td.ghost_age[i] # Convert to a positive number

            # Scale pixel intensities
            lab_frame_image = scale_image(freeimage.read(files[worm_name][image_timepoint][0]))
            # Get spline information
            center_tck, width_tck = positions[worm_name][1][image_timepoint]['pose']

            if center_tck is not None and width_tck is not None:
                # Get worm-frame image, and process it
                worm_frame_image = standardized_worm_frame_image(lab_frame_image, center_tck, width_tck, 86, 720)
                new_worm_image = create_RGB_stack(square_worm_frame(worm_frame_image))
                outfile_name = worm_name + '_' + image_timepoint + '.png'
                freeimage.write(new_worm_image, OUTPUT_DIRECTORY + outfile_name)

                worm_image_data.append([worm_name, image_timepoint, worm_lifespan, hours_remaining])


# Save image-specific worm data to a csv file for futre analyses
with open(OUTPUT_DIRECTORY + 'image_data.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for sublist in worm_image_data:
        wr.writerow(sublist)

