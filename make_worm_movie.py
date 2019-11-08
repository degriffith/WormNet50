# make_worm_movie.py

from elegant import worm_data
from elegant import load_data
from elegant import process_images
from elegant import worm_spline
from elegant.gui import compose_images
from ris_widget import ris_widget
from zplib import curve
from zplib.image import write_movie
import pickle
import freeimage 
import numpy as np
import csv

"""
@filename: make_worm_movie.py
@authors: Dan Griffith, Zach Pincus (zplab)
@project: CNN for deep phenotypic analysis of C. elegans -- DG Rotation project

Usage:
Makes a movie of a worm over the course of its life. Currently outputs as a movie worm-frame images,
but can be modified to create a lab-frame image movie.

User-specified parameters:
--NAME: which worm should the movie be made for
--experimental_root: root folder with images/data
--worm_annotation_files: annotation .tsv files
--OUTPUT_DIR: where the movie should be saved

"""

############################# Functions ###########################

def to_tck(widths):
       x = np.linspace(0, 1, len(widths))
       smoothing = 0.0625 * len(widths)
       return curve.interpolate.fit_nonparametric_spline(x, widths, smoothing=smoothing)

# Adjust worm frame widths and lengths to standard values
WIDTH_TRENDS = pickle.load(open('/Users/zplab/miniconda/lib/python3.7/site-packages/elegant/width_data/width_trends.pickle', 'rb'))
# Widths of average worm at the age of 5 days (interpolated)
AVG_WIDTHS = np.array([np.interp(5, WIDTH_TRENDS['ages'], wt) for wt in WIDTH_TRENDS['width_trends']])
AVG_WIDTHS_TCK = to_tck(AVG_WIDTHS)

def standardized_worm_frame_image(lab_frame_image, center_tck, width_tck, pixel_height, 
	    pixel_width, average_widths_tck=AVG_WIDTHS_TCK):
    WORM_WIDTH= 64
    WORM_PAD = 38

    worm_width_factor = (pixel_height - WORM_PAD) / WORM_WIDTH
    new_avg_width_tck = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1] * worm_width_factor, AVG_WIDTHS_TCK[2])
    return worm_spline.to_worm_frame(lab_frame_image, center_tck, width_tck, 
    	standard_length=pixel_width, standard_width=new_avg_width_tck)


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
    return image

################################################################

# User-specified parameters
NAME = "06"
experimental_root =  '/Volumes/9karray/Mosley_Matt/20190408_lin-4_spe-9_20C_pos-1'
worm_annotation_files = '/Volumes/9karray/Mosley_Matt/20190408_lin-4_spe-9_20C_pos-1/derived_data/measurements/core_measures/*.tsv'
OUTPUT_DIR = '/Users/zplab/Desktop/DanScripts/MiscFigures/'


# Open worm images and annotation information
rw = ris_widget.RisWidget()
worm_annotations = worm_data.read_worms(worm_annotation_files, name_prefix='')
worm_annotations.sort('name') # Optional sort, but easier to follow for debugging
files = load_data.scan_experiment_dir(experimental_root)

# Image annotation data
positions = load_data.read_annotations(experimental_root)
positions = load_data.filter_annotations(positions, load_data.filter_excluded)

# Iterate through all worms, select worm of interest
for worm in worm_annotations:
    worm_name = worm.name
    # only select the correct worm
    if worm_name != NAME:
    	continue

    for i in range(len(worm.td.timepoint)):
    	# Ensure that worm is an adult (can use other filter depending on dataset)
        if worm.td.stage[i] == 'adult':
        	image_timepoint = worm.td.timepoint[i]

            # Convert scope image to worm frame image
        	lab_frame_image = freeimage.read(files[worm_name][image_timepoint][0])
        	lab_frame_image = scale_image(lab_frame_image)
        	center_tck, width_tck = positions[worm_name][1][image_timepoint]['pose']
        	worm_frame_image = standardized_worm_frame_image(lab_frame_image, center_tck, width_tck, 85, 720)
        	
            # Add worm image to flipbook
        	rw.flipbook_pages.append(worm_frame_image)

# Create movie and save to file
image_generator = compose_images.generate_images_from_flipbook(rw)
write_movie.write_movie(image_generator, OUTPUT_DIR + 'Worm_' + NAME + '_movie.mp4', framerate=8)



