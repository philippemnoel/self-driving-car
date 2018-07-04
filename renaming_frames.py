# Philippe M. Noel
# Renaming jpg file names for self-driving training purposes
# This file was used to rename the Udacity dataset to fit the Sully Chen
# dataset format
import os

def renaming(directory):
	""" Renames all the files in directory to from 0.jpg to (number of files - 1).jpg """
	# initialize file counter
	fc = 0
	# rename every file in the directory
	for file in sorted(os.listdir(directory)):
		# rename the file & update counter
		os.rename(os.path.join(directory, file), os.path.join(directory, str(fc) + '.jpg'))
		fc += 1

# function call
renaming(os.path.abspath('/Users/noep/Desktop/Autopilot-TensorFlow-master/Ch2_001.tar/center/'))
