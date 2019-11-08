# loess_prediction.py

import pandas as pd 
import numpy as np
from scipy import stats
from skmisc.loess import loess
from scipy.interpolate import interp1d
import math

"""
@filename: loess_prediction.py
@authors: Dan Griffith, Zach Pincus (zplab)
@project: CNN for deep phenotypic analysis of C. elegans -- DG Rotation project

Usage:
Calculate a locally-weighted least squares polynomical regression (LOESS) on an individual worm's 
regression predictions over its lifetime. 

There are two options when using this script. The top section allows one to plot the regression 
predictions with the LOESS curve and a 95% CI for individual worm. The bottom section is designed
to calculate and plot the average LOESS curve of the lifespan quintiles.

User-specified parameters:
--csv_dir
--csv_files
--TARGET_X: the x-value of the plot, AKA the true value of the regression target
--TARGET_Y: the target value that is being regressed

"""

######################### User-specified paramenters ############################

csv_dir = '/mnt/lugia_array/Griffith_Dan/worm_images/all_images_mask/'
csv_files = ['matt0v_image_data.csv', 'matt1v_image_data.csv', 'matt2v_image_data.csv', 
                        'matt3v_image_data.csv', 'matt4v_image_data.csv']
TARGET_X = 'hours_remaining'
TARGET_Y = 'combo_hours'

#################################################################################
# Compute a LOESS trace for an individual worm's scatterplot and calculate the 95% confidence interval
# of this trace using non-parametric bootstrapping

"""
num_file = 4
csv_filename = csv_dir + csv_files[num_file]
df = pd.read_csv(csv_filename, usecols=[1,2,3,4,5,6,7,8,9,10], dtype={'name':'str'})
worms = pd.unique(df['name'])

####

for i in range(len(worms)):
	name = worms[i]
	worm_df = df[df['name'] == name]
	print(name)

	# x, y pairs, sort by x
	points = np.vstack((worm_df[TARGET_X].values, worm_df[TARGET_Y].values))
	points = points.T
	points = points[points[:,0].argsort()].T

	# Calculate LOESS
	l = loess(points[0], points[1], family='symmetric', span=0.4, normalize=False)
	l.fit()
	pred = l.predict(points[0], stderror=True)
	conf = pred.confidence()
	lowess = pred.values

	n = len(points[0])
	bootstrap_dict = {}
	for j in range(n):
		bootstrap_dict[j] = np.array([])

	# Resample points with replacement 10,000 times and recompute LOESS curve
	for k in range(10000):
		resample_idx = np.random.randint(0,n,n)
		resample_idx.sort()
		resample = np.array([[0,0]])
		for j in range(n):
			resample = np.append(resample, [points.T[resample_idx[j]]], axis=0)
		resample = resample[1:,:].T

		l_bootstrap = loess(resample[0], resample[1], family='symmetric', span=0.4, normalize=False)
		l_bootstrap.fit()
		pred_bootstrap = l_bootstrap.predict(resample[0])
		bootstrap_lowess = pred_bootstrap.values

		for j in range(n):
			bootstrap_dict[resample_idx[j]] = np.append(bootstrap_dict[resample_idx[j]], bootstrap_lowess[j])

	# Take the bottom and top 2.5% LOESS prediction value of each point as the 95% CI
	upperbounds = []
	lowerbounds = []
	for x_idx in range(n):
		bootstrap_dict[x_idx].sort()
		m = len(bootstrap_dict[x_idx])
		upper = math.ceil(m * .975) 
		lower = math.floor(m * 0.025)

		upperbounds.append(bootstrap_dict[x_idx][upper])
		lowerbounds.append(bootstrap_dict[x_idx][lower])

	# Smooth the CI line
	x_new = np.linspace(points[0,0], points[0,-1], 500)
	f_upper = interp1d(points[0], upperbounds, kind='quadratic')
	f_lower = interp1d(points[0], lowerbounds, kind='quadratic')
	ul=f_upper(x_new)
	ll=f_lower(x_new)

	from matplotlib import pyplot as plt

	plt.figure(i, figsize=(8.0, 6.0))
	plt.plot(points[0], points[1], '+')
	plt.plot(points[0], lowess)
	plt.fill_between(x_new, ll, ul, alpha=0.33)
	
	# Comment out 2/3 below based on which target value is being used
	plt.title("True vs predicted hours remaining: Worm " + name, fontsize=16)
	plt.xlabel("True hours remaining")
	plt.ylabel("Predicted hours remaining")
	plt.xlim(left=0, right=800)
	plt.ylim(bottom=0, top=800)
	plt.plot([0, 800], [0, 800], 'k--', linewidth=0.5)
	'''
	plt.title("True vs predicted percent remaining: Worm " + name, fontsize=16)
	plt.xlabel("True percent remaining")
	plt.ylabel("Predicted percent remaining")
	plt.xlim(left=0, right=100)
	plt.ylim(bottom=0, top=100)
	plt.plot([0, 100], [0, 100], 'k--', linewidth=0.5)
	
	plt.title("True vs predicted age: Worm " + name, fontsize=16)
	plt.xlabel("True age (days)")
	plt.ylabel("Predicted age (days)")
	plt.xlim(left=0, right=800/24)
	plt.ylim(bottom=0, top=800/24)
	plt.plot([0, 800/24], [0, 800/24], 'k--', linewidth=0.5)
	'''
	
	
	plt.savefig('/mnt/lugia_array/Griffith_Dan/figures/LOESS_figures/--worm' + name + '_comboHours_loess.png')
	plt.show()

"""
######################################

# Sort worms into lifespan quintiles and find best fit for the combined
# regression of an entire quintile
quintile1_names = ['02','17','19','26','35','41','43','46','52','53','54','55']
quintile2_names = ['04','05','09','18','25','51','56','58','61','63','65']
quintile3_names = ['00','14','21','24','27','30','34','36','38','44','49','66']
quintile4_names = ['06','11','13','31','40','42','48','50','57','59','62']
quintile5_names = ['01','03','08','15','16','29','32','33','39','45','47','64']

quintile1_points = np.array([[0],[0]])
quintile2_points = np.array([[0],[0]])
quintile3_points = np.array([[0],[0]])
quintile4_points = np.array([[0],[0]])
quintile5_points = np.array([[0],[0]])


for num_file in range(5):
	print('\tnum_file=', num_file)
	csv_filename = csv_dir + csv_files[num_file]
	df = pd.read_csv(csv_filename, usecols=[1,2,3,4,5,6,7,8,9,10], dtype={'name':'str'})
	worms = pd.unique(df['name'])

	for i in range(len(worms)):
		name = worms[i]
		worm_df = df[df['name'] == name]

		# x, y pairs, sort by x
		points = np.vstack((worm_df[TARGET_X].values, worm_df[TARGET_Y].values))
		points = points.T
		points = points[points[:,0].argsort()].T

		# Calculate LOESS
		l = loess(points[0], points[1], family='symmetric', span=0.4, normalize=False)
		l.fit()
		pred = l.predict(points[0], stderror=True)
		lowess = pred.values

		# Sort into quintiles
		if name in quintile1_names:
			print(name, "-> 1st quintile")
			quintile1_points = np.hstack((quintile1_points, np.vstack((points[0], lowess))))
		elif name in quintile2_names:
			print(name, "-> 2nd quintile")
			quintile2_points = np.hstack((quintile2_points, np.vstack((points[0], lowess))))
		elif name in quintile3_names:
			print(name, "-> 3rd quintile")
			quintile3_points = np.hstack((quintile3_points, np.vstack((points[0], lowess))))
		elif name in quintile4_names:
			print(name, "-> 4th quintile")
			quintile4_points = np.hstack((quintile4_points, np.vstack((points[0], lowess))))
		elif name in quintile5_names:
			print(name, "-> 5th quintile")
			quintile5_points = np.hstack((quintile5_points, np.vstack((points[0], lowess))))

# Sort the matrix by the true values
quintile1_points = quintile1_points[:, 1:].T
quintile1_points = quintile1_points[quintile1_points[:,0].argsort()].T

quintile2_points = quintile2_points[:, 1:].T
quintile2_points = quintile2_points[quintile2_points[:,0].argsort()].T

quintile3_points = quintile3_points[:, 1:].T
quintile3_points = quintile3_points[quintile3_points[:,0].argsort()].T

quintile4_points = quintile4_points[:, 1:].T
quintile4_points = quintile4_points[quintile4_points[:,0].argsort()].T

quintile5_points = quintile5_points[:, 1:].T
quintile5_points = quintile5_points[quintile5_points[:,0].argsort()].T

##################### LOESS curves for average quintile loess ########################

# Quintile 1:
quint1_l = loess(quintile1_points[0], quintile1_points[1], degree=2, family='symmetric', span=0.15, normalize=False)
quint1_l.fit()
quint1_pred = quint1_l.predict(quintile1_points[0], stderror=True)
quint1_conf = quint1_pred.confidence()

quint1_lowess = quint1_pred.values
quint1_ll = quint1_conf.lower
quint1_ul = quint1_conf.upper

# Quintile 2:
quint2_l = loess(quintile2_points[0], quintile2_points[1], degree=2, family='symmetric', span=0.15, normalize=False)
quint2_l.fit()
quint2_pred = quint2_l.predict(quintile2_points[0], stderror=True)
quint2_conf = quint2_pred.confidence()

quint2_lowess = quint2_pred.values
quint2_ll = quint2_conf.lower
quint2_ul = quint2_conf.upper

# Quintile 3:
quint3_l = loess(quintile3_points[0], quintile3_points[1], degree=2, family='symmetric', span=0.15, normalize=False)
quint3_l.fit()
quint3_pred = quint3_l.predict(quintile3_points[0], stderror=True)
quint3_conf = quint3_pred.confidence()

quint3_lowess = quint3_pred.values
quint3_ll = quint3_conf.lower
quint3_ul = quint3_conf.upper

# Quintile 4:
quint4_l = loess(quintile4_points[0], quintile4_points[1], degree=2, family='symmetric', span=0.15, normalize=False)
quint4_l.fit()
quint4_pred = quint4_l.predict(quintile4_points[0], stderror=True)
quint4_conf = quint4_pred.confidence()

quint4_lowess = quint4_pred.values
quint4_ll = quint4_conf.lower
quint4_ul = quint4_conf.upper

# Quintile 5:
quint5_l = loess(quintile5_points[0], quintile5_points[1], degree=2, family='symmetric', span=0.15, normalize=False)
quint5_l.fit()
quint5_pred = quint5_l.predict(quintile5_points[0], stderror=True)
quint5_conf = quint5_pred.confidence()

quint5_lowess = quint5_pred.values
quint5_ll = quint5_conf.lower
quint5_ul = quint5_conf.upper

#Plot each quintile's loess curve with it's standard error 

from matplotlib import pyplot as plt
plt.figure(0, figsize=[8,6])

# Quintile 1:
#plt.plot(quintile1_points[0], quintile1_points[1], '+')
plt.plot(quintile1_points[0], quint1_lowess)
plt.fill_between(quintile1_points[0], quint1_ll, quint1_ul, alpha=0.33)
# Quintile 2:
plt.plot(quintile2_points[0], quint2_lowess)
plt.fill_between(quintile2_points[0], quint2_ll, quint2_ul, alpha=0.33)
# Quintile 3:
plt.plot(quintile3_points[0], quint3_lowess)
plt.fill_between(quintile3_points[0], quint3_ll, quint3_ul, alpha=0.33)
# Quintile 4:
plt.plot(quintile4_points[0], quint4_lowess)
plt.fill_between(quintile4_points[0], quint4_ll, quint4_ul, alpha=0.33)
# Quintile 5:
plt.plot(quintile5_points[0], quint5_lowess)
plt.fill_between(quintile5_points[0], quint5_ll, quint5_ul, alpha=0.33)



plt.title("True vs predicted hours remaining: lifespan quintiles", fontsize=16)
plt.xlabel("True hours remaining")
plt.ylabel("Predicted hours remaining")
plt.xlim(left=0, right=800)
plt.ylim(bottom=0, top=800)
plt.plot([0, 800], [0, 800], 'k--', linewidth=0.5)
'''
plt.title("True vs predicted percent remaining: lifespan quintiles", fontsize=16)
plt.xlabel("True percent remaining")
plt.ylabel("Predicted percent remaining")
plt.xlim(left=0, right=100)
plt.ylim(bottom=0, top=100)
plt.plot([0, 100], [0, 100], 'k--', linewidth=0.5)

plt.title("True vs predicted age: lifespan quintiles", fontsize=16)
plt.xlabel("True age (days)")
plt.ylabel("Predicted age (days)")
plt.xlim(left=0, right=800/24)
plt.ylim(bottom=0, top=800/24)
plt.plot([0, 800/24], [0, 800/24], 'k--', linewidth=0.5)
'''


plt.savefig('/mnt/lugia_array/Griffith_Dan/figures/LOESS_figures/----')
