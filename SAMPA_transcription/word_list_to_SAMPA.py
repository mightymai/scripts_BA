import epitran  
import os
import pandas
import numpy


########### ADJUST ###########
folder_path = '/home/mai/BA_wuerdig/data_100/test_chars'
##############################

path = folder_path + '/dataset/train/audio'
FolderList = [folder for folder in os.listdir(path) if ("." not in folder) and ("_" not in folder) and ("9" not in folder)]

epi = epitran.Epitran('eng-Latn') 

words = []
chars = set()
for folder in FolderList:
	word_temp = epi.xsampa.ipa2xs(epi.transliterate(folder))
	words.append(word_temp)
	chars.update(list(word_temp))

chars = list(sorted(chars))

value = ''
chars = [value] + chars
chars.append('9')
chars.append('_')

data1 = pandas.DataFrame(words, columns =['words_SAMPA'])
data2 = pandas.DataFrame(FolderList, columns =['labels'])
data3 = pandas.DataFrame(chars, columns =['chars_SAMPA'])
data4 = list(range(0,len(data3)))
data4 = pandas.DataFrame(data4, columns =['chars_len'])

merged_df = pandas.concat([data2, data1], axis = 1)
merged_df_2 = pandas.concat([data4, data3], axis = 1)

numpy.savetxt(folder_path + '/map_words.txt', merged_df.values, fmt='%s', delimiter=',')
numpy.savetxt(folder_path + '/map_chars.txt', merged_df_2.values, fmt='%s', delimiter=',')

print('finished')