import os
import numpy as np
import shutil

############## ADJUST ##############
freq = 20
path = '/home/mai/BA_wuerdig/data_20/complete_audio_20/unbalanced'
balanced = "/home/mai/BA_wuerdig/data_20/complete_audio_20/balanced"
word_list = [folder for folder in os.listdir(path) if ("." not in folder) and ("LICENSE" not in folder) and ("_background_noise_" not in folder) and ("9" not in folder)]
####################################

print(word_list)

for word in word_list:
	counter = 0
	already_copied = []
	source_path = path + "/" + word + "/"
	files = os.listdir(source_path)
	os.mkdir(balanced + "/" + word + "/")
	dest_path = balanced + "/" + word + "/"
	if len(os.listdir(dest_path)) < freq:
		while counter < freq:
			file_to_copy = files[np.random.randint(0,len(files))]
			if file_to_copy not in already_copied:
				shutil.copyfile(source_path + file_to_copy, dest_path + file_to_copy)
				print(file_to_copy, "copied")
				counter += 1
			already_copied.append(file_to_copy)
			#files.remove(file_to_copy)

	else:
		print("directory already contains files")