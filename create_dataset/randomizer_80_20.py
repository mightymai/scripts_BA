import os
import numpy as np
import shutil
import distutils
from distutils.dir_util import copy_tree 

path = '/home/mai/BA_wuerdig/data_100/complete_audio_100/balanced'
dat_80_path = '/home/mai/BA_wuerdig/data_100/balanced/dataset/train/audio'
dat_20_path = '/home/mai/BA_wuerdig/data_100/balanced/dataset/test/audio'
word_list = [folder for folder in os.listdir(path) if ("." not in folder) and ("LICENSE" not in folder) and ("_background_noise_" not in folder) and ("9" not in folder)]


for word in word_list:
	counter = 1
	already_copied = []
	source_path = path + "/" + word + "/"
	files = os.listdir(source_path)
	os.mkdir(dat_80_path + "/" + word + "/")
	dest_path = dat_80_path + "/" + word + "/"
	if len(os.listdir(dest_path)) <= (len(os.listdir(source_path)) * 0.8):
		while counter <= (len(os.listdir(source_path)) * 0.8):
			file_to_copy = files[np.random.randint(0,len(files)-1)]
			if file_to_copy not in already_copied: 
			 	shutil.copyfile(source_path + file_to_copy, dest_path + file_to_copy)
			 	print(file_to_copy, "copied, word ", dest_path)
			 	counter += 1
			already_copied.append(file_to_copy)
	else:
		print("directory already contains files")

print('copying data (might take some time)')
distutils.dir_util.copy_tree(path, dat_20_path)  
print('finished copying') 

word_list = [folder for folder in os.listdir(dat_20_path) if ("." not in folder) and ("LICENSE" not in folder) and ("_background_noise_" not in folder) and ("9" not in folder)]

for word in word_list:
	new_path = dat_20_path + "/" + word + "/"
	test_files = os.listdir(new_path)

	train_source_path = dat_80_path + "/" + word + "/"
	train_files = os.listdir(train_source_path)

	complete_files = test_files + train_files

	doubles = set([x for x in complete_files if complete_files.count(x) > 1])

	for file in doubles:
		print("Removing "  + str(file))
		os.remove(os.path.join(new_path, file))

print('finished creating test and train data')