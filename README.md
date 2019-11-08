Authors: Dan Griffith, Zach Pincus (zplab)

Project: CNN for deep phenotypic analysis of C. elegans -- DG Rotation project

The files in this repository are intended to be used in a C. elegans deep learning phenotypical 
analysis pipeline using microscope images and a convolutional neural network (CNN). The general 
pipeline is as follows:


Aggregate and process images -> Split into train/val data -> Build CNN and regression -> Analysis


If starting on a brand new worm image dataset, a typical workflow using these scripts would 
proceed as such:

1. aggregate_and_preprocess_images.py

	User needs to set the directory in which the images and annotation data are located. This
	script will take the annotated scope images and return a processed worm-frame image with
	standardized shape and a environment mask. These new images can be saved to a new directory
	for later steps in the pipeline.

	Additionally, this script will output a csv file containing pertinent image-specific worm
	information. This data is essential for later regression and analysis.

2. divide_test_by_worm.py (most likely copy and paste code into ipython)

	After the images are processed, they need to be split into training and validation sets
	before the network can be trained. It is HIGHLY recommended that training/validation sets
	are split BY WORM. That is to say that a subset of worms, and ALL of their corresponding
	timepoint images make up the validation data. Then the images in the validation set will
	all come from worms that have never been seen during training, which will help prevent 
	overfitting.

	This script allows one to split the data two-fold, directly into train/val sets 
	if there are enough images. Alternatively, if the dataset is smaller it can be split
	five-fold, for later cross-validated training/testing. In either case, the split data
	will also have corresponding split csv files.

3. WormNet50_regressor.py (or classifier)

	This script is where the meat of deep learning and regression takes place. Here, the CNN is 
	constructed and validated using PyTorch. The user specifies what data to regress on, what the 
	learning parameters are, what the target value is, along with other possible features 
	depending on what analysis will be conducted. This script is run from terminal, and can 
	produce an informative regression scatterplot and training progression plot. In addition, the 
	weights from the best-performing network can be saved and used later.

	Alternatively, if the user is interested in a classification task rather than regression, that
	can be accomplished using a slightly different set up and script. Refer to in-script 
	documentation (or the jupyter notebook pages) for additional information on regression and
	classification.

4. age_bin_cross_val.py

	This is typically the most important script for analysis, as it allows one to combine the 
	results of 5-fold cross-validation regression into a single scatterplot. Additionally, it
	can provide informative information on predictive performance in relationship to worm age.

5. Further analysis

	The remaining scripts are for specialized analysis. Briefly:

	test_individual_worm.py -- Create single worm scatterplot; add regression prediction to csv file
	
	loess_prediction.py -- Single worm scatterplot + LOESS curve; averaged LOESS quintiles
	
	make_worm_movie.py -- Create a timelapse movie of a single worm using the worm-frame perspective
	

For additional information about this pipeling and previous findings, refer to the powerpoint in
lugia_array in the folder /Griffith_Dan/figures/presentation/


