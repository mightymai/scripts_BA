import os
import pandas
import csv

path = os.path.dirname(os.path.realpath(__file__))
FolderList = [folder for folder in os.listdir(path) if ("." not in folder) and ("_" not in folder) and ("9" not in folder)]


big_folder_data = []
big_folder_words = []

for folder in FolderList:
	folder_data = []
	folder_words = []
	folder_path = path + "/" + folder + "/"
	folder_data = os.listdir(folder_path)
	num_words = len(folder_data)

	i = 1
	while i <= num_words:
		folder_words.append(folder)
		i = i + 1

	big_folder_data.extend(folder_data)
	big_folder_words.extend(folder_words)



dataframeW = pandas.DataFrame(big_folder_words, columns =['word'])
dataframeD = pandas.DataFrame(big_folder_data, columns =['dat'])


merged_df = pandas.concat([dataframeD, dataframeW], axis = 1)

csv_final = merged_df.to_csv('folder_list.csv', sep='\t', index = False)


