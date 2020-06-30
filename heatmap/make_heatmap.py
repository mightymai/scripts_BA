import subprocess
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import Levenshtein as L
import epitran  
import math

# this function sorts the confusion matrix (data) by the frequency of the words' occurences. 
def sort_csv(data, submission, dat_val, heatmap_title, output_name, number, current,  colorbar_bool, folder_list='folder_list'):
	########## putting dataframes into wanted form ##########
	table = data + '.csv'
	dat = pd.read_csv(table, index_col=0) 
	dat_S = dat

	Summe = dat_S['Summe'].tolist()
	dat_N = maybe_drop_colum(dat_S, ['Summe']) 


	index_dat_S = []
	for row in dat_S.index:  
	     index_dat_S.append(row)

	name = index_dat_S
	dat_S['labels'] = name

	##### für nach Häufigkeit sortierte zweite heatmap #####
	new_table = dat_S.sort_values(by=['Summe', 'labels'], ascending=False)
	new_table = maybe_drop_colum(new_table, ['labels', 'Summe']) 

	new_table = new_table.transpose()

	Summe.append(0)
	new_table['Summe'] = Summe
	name.append('z')
	new_table['name'] = name
	 
	new_table = new_table.sort_values(by=['Summe', 'name'], ascending=False)

	new_table = maybe_drop_colum(new_table, ['name', 'Summe']) 

	new_table = new_table.transpose()

	output = make_heatmap(new_table, submission, dat_val, heatmap_title, output_name, number, current, colorbar_bool, folder_list)

	return output

	#########################################################

# this function sorts the confusion matrix (dat_10) by the frequency of the words' occurences of another confusion matrix (dat_5)
def sort_csv_alien(dat_5, dat_10, dat_10_val, submission, heatmap_title, output_name, number, current, colorbar_bool, folder_list):
	dat_50 = pd.read_csv(dat_5 + '.csv', index_col=0) 
	dat_100 = pd.read_csv(dat_10 + '.csv', index_col=0) 

	dat_50 = maybe_drop_colum(dat_50, ['Diag', 'Diag_p']) 

	################### get frequencies from other file ###################
	Summe = dat_50['Summe'].tolist()

	index_dat_50 = []
	for row in dat_50.index:  
	     index_dat_50.append(row)

	index_dat_100 = []
	for row in dat_100.index:  
	     index_dat_100.append(row)

	data_tuples = list(zip(index_dat_50, Summe))   
	df_50 = pd.DataFrame(data_tuples, columns=['words','Summe'])  

	range_d = len(df_50) - 1

	while (range_d != -1):
		cell = df_50.iloc[range_d]['words']
		if(cell not in index_dat_100):
			df_50 = df_50[~(df_50['words'] == cell)]  
			#print(cell + ' was dropped, ' + '50: ' + str(len(df_50)) + ', 100: ' + str(len(index_dat_100)) + ', number is: ' + str(range_d))
			#else: print('50: ' + str(len(df_50)) + ', 100: ' + str(len(index_dat_100))  + ', number is: ' + str(range_d))
		range_d -= 1

	# df with words and freqs
	df_50

	########################################################################

	############################## sortieren ###############################

	list_freq = df_50['Summe'].tolist()
	df_100 = pd.DataFrame(dat_100)
	df_100 = maybe_drop_colum(df_100, ['Summe']) 

	index_df_100 = []
	for row in df_100.index:  
		index_df_100.append(row)

	# Spalten sortieren
	df_100['Summe'] = list_freq
	df_100['labels'] = index_df_100

	df_100 = df_100.sort_values(by=['Summe', 'labels'], ascending=False)
	df_100 = maybe_drop_colum(df_100, ['labels', 'Summe']) 

	df_100 = df_100.transpose()

	# Reihen sortieren
	list_freq.append(0)
	index_df_100.append('z')
	df_100['Summe'] = list_freq
	df_100['name'] = index_df_100
	df_100 = df_100.sort_values(by=['Summe', 'name'], ascending=False)
	df_100 = maybe_drop_colum(df_100, ['name', 'Summe']) 

	df_100 = df_100.transpose()

	output = make_heatmap(df_100, submission, dat_10_val, heatmap_title, output_name, number, current, colorbar_bool, folder_list)

	return output
	########################################################################


# this function makes a heatmap for a given confusion matrix (ojdata) and adds the mean an overall accuracy
def make_heatmap(ojdata, submission, dat_val, heatmap_title, output_name, number, current, colorbar_bool, folder_list):
	#plt.subplot(1,number,current)
	dat = pd.DataFrame(ojdata.values)
	minimum = dat.min()
	maximum = dat.max()
	#plt.axis('equal')
	#print(minimum)
	#print(maximum)
	sn.set(font_scale=0.8)
	# schöne Farben: viridis, cool, RdPu, magma, Spectral, gist_stern, gnuplot
	# xticklabels=['actual'], yticklabels=['predicted']
	ax = plt.axes()
	#ax = fig.add_subplot(111, aspect='equal')
	cbar_kws = {'ticks': [0.0, 0.25, 0.5, 0.75, 1.0], # set ticks of color bar
	            'label':'Percentage of correct Predictions'}

	if(colorbar_bool):
		plt.tight_layout()
		dat_new = sn.heatmap(dat, annot=False, xticklabels=False, yticklabels=False, cmap='gnuplot', mask=(dat==0), cbar_kws = cbar_kws, ax = ax, square=True)# font size
		dat_new.set_xlabel("X Label",fontsize=15)
		dat_new.set_ylabel("Y Label",fontsize=15)
		plt.subplots_adjust(top = 0.93, bottom=0.055, left=0.01)
	else:
		plt.tight_layout()
		dat_new = sn.heatmap(dat, annot=False, xticklabels=False, yticklabels=False, cmap='gnuplot', mask=(dat==0), ax = ax, cbar = False, square=True)# font size
		dat_new.set_xlabel("X Label",fontsize=15)
		dat_new.set_ylabel("Y Label",fontsize=15)
		plt.subplots_adjust(top = 0.93, bottom=0.055, left=0.01)

	ax.figure.axes[-1].yaxis.label.set_size(15)
	dat_new.set_xlabel('predicted',fontsize=15)
	dat_new.set_ylabel('actual',fontsize=15)

	figure = dat_new.get_figure() 
	ax.set_title(heatmap_title, fontsize=20)
	#figure.show()
	####################################   

	########### 1. accuracy ############
	#data = pd.read_csv(new_table) 
	data = ojdata   

	#data = pd.read_csv(ojdata)    
   
	acc = compute_acc(data) 

	####################################

	########### 2. accuracy ############
	sub_data = pd.read_csv(submission)   

	second_table = dat_val + '.csv'
	data_2 = pd.read_csv(second_table, index_col=0)  
	data_2 = maybe_drop_colum(data_2, ['Diag', 'Diag_p', 'Summe'])   
	
	acc_2 = compute_acc(data_2, sub_data) 

	####################################

	#per = compute_per(folder_list, submission) 

	''' NOT USED FOR FINAL PLOT
	if(colorbar_bool):
		unknown_per = unknown_percent(second_table)
		mean_acc_text = ax.text(0, -0.075, str('mean accuracy = ' + str(acc)), fontsize=8, transform=ax.transAxes)
		all_acc_text = ax.text(0, -0.125, str('overall accuracy = ' + str(acc_2)), fontsize=8, transform=ax.transAxes)
		unknown_per_text = ax.text(1, -0.075, str('unknown = ' + str(unknown_per)), fontsize=8, transform=ax.transAxes, horizontalalignment='right')
	else:
		unknown_per = unknown_percent(second_table)
		mean_acc_text = ax.text(0, -0.075, str('mean accuracy = ' + str(acc)), fontsize=8, transform=ax.transAxes)
		all_acc_text = ax.text(0, -0.125, str('overall accuracy = ' + str(acc_2)), fontsize=8, transform=ax.transAxes)
		unknown_per_text = ax.text(1, -0.075, str('unknown = ' + str(unknown_per)), fontsize=8, transform=ax.transAxes, horizontalalignment='right')
	'''

	figure.savefig(output_name + '.png', dpi=400)

	''' NOT USED FOR FINAL PLOT
	unknown_per_text.set_visible(False)
	mean_acc_text.set_visible(False)
	all_acc_text.set_visible(False)
	'''

def compute_per(folder_list_o, submission_o):

	folder_list = folder_list_o + '.csv'
	folder_list = pd.read_csv(folder_list, sep='\t')

	submission = submission_o[:10] + '_dbg' + submission_o[10:]
	submission = pd.read_csv(submission)

	# sichergehen, dass gleich sortiert
	folder_list = folder_list.sort_values(by=['dat'])
	submission = submission.sort_values(by=['fname'])

	word = folder_list['word'].tolist()
	submission['word_label'] = word
	unique_word = set(word)
	u_word = pd.DataFrame(unique_word, columns=['label'])
	u_word['SAMPA'] = ""

	# Translation to SAMPA
	epi = epitran.Epitran('eng-Latn') 

	words = []
	for row in u_word['label']:
		word_temp = epi.xsampa.ipa2xs(epi.transliterate(row))
		u_word.SAMPA[u_word.label==row] = word_temp

	folder_list = folder_list.merge(u_word, how='left', left_on='word', right_on='label') 
	folder_list = folder_list.drop(['label'], axis=1)     

	submission = submission.merge(folder_list, how='left', left_on='fname', right_on='dat') 
	submission = submission.drop(['dat', 'word_label'], axis=1)     

	submission['LD'] = ""
	submission['seq_len'] = ""
	submission['PER'] = ""

	for row in range(len(submission)):
		if (row%1000 == 0):
 			print('current row ' + str(row) + ' of ' + str(len(submission)))

		pred_temp = submission.iloc[row,3]
		SAMPA_temp = submission.iloc[row,5]
		if(isNaN(pred_temp)):
			LD_temp = len(submission.iloc[row,4])
		else: 
			LD_temp = L.distance(pred_temp, SAMPA_temp)
		submission.iloc[row,6] = LD_temp

		seq_len_temp = len(SAMPA_temp)
		submission.iloc[row,7] = seq_len_temp

		PER_temp = LD_temp / seq_len_temp
		submission.iloc[row,8] = PER_temp


	PER_list = submission['PER'].tolist()

	PER_sum = sum(PER_list)

	sum_of_tokens = len(submission)

	PER = PER_sum / sum_of_tokens

	PER = format(PER, '.4f')
	return PER


def isNaN(string):
    return string != string


def compute_acc(data, submission = False):
	summe = 0 
	for rownum in range(len(data) - 1):
		summe = summe + data.iloc[rownum][rownum]

	if (isinstance(submission, bool)):
		acc = summe / (len(data)) 
	else: 
		acc = summe / (len(submission)) 
	acc = format(acc, '.4f')
	return acc

def maybe_drop_colum(dat, col):
	for name in col:
		if name in dat.columns:
			dat = dat.drop([name], axis = 1)
	return dat

def unknown_percent(dat_v):
	data = pd.read_csv(dat_v)
	list_sum = data.sum(axis=0)
	unknown = list_sum['unknown']
	Summe = list_sum['Summe']
	return format(unknown/Summe, '.4f')


def call_Rscript(folder_list, sub, output_name):
	subprocess.call(['Rscript', 'make_confusion_matrix.R', str(folder_list) + '.csv', 'submission_0_0-' + str(sub)  + '.csv', str(output_name)])

def yes_no(question):
    while "invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        list = ['y', 'yes', 'n', 'no']
        if (reply[:1] in list):
            return reply[:1]

def yes_no_bool(question):
    while "invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        list = ['y', 'yes', 'n', 'no']
        if (reply[:1] == 'y' or reply[:1] == 'yes'):
        	return True
        elif (reply[:1] == 'n' or reply[:1] == 'no'):
        	return False


def check_existence_csv(file):
	try: 
		x = open(file) 
		print('\t   ✓ Specified file is avaiable.')
		x.close() 
		return(file[:-4])
	except IOError: 
		print('\t   ✗ Specified file is not avaiable.')
		response = str(input('\t     Please try again: '))
		check_existence_csv(response + '.csv')
		return response

def check_existence_sub(file):
	try: 
		x = open(file) 
		print('\t   ✓ Specified file is avaiable.')
		x.close() 
		return(file[15:-4])
	except IOError: 
		print('\t   ✗ Specified file is not avaiable.')
		response = str(input('\t     Please try again: '))
		check_existence_sub(to_sub_format(response))
		return response


def to_sub_format(sub):
	new_sub = 'submission_0_0-' + str(sub)  + '.csv'
	return new_sub

def abfrage():
	print("\n")
	print('Don\'t add the file\'s suffix, only the filename. ')
	conf = yes_no('Do you want to create a confusion matrix?')
	if (conf == 'y' or conf == 'yes'):
		print('For creating the confusion matrix enter: ')
		arg1 = str(input('	- filename of the folder list: '))
		arg1 = check_existence_csv(arg1 + '.csv')
		f_list = arg1
		arg2 = str(input('	- submission (number of steps) and optional endings: '))
		arg2 = check_existence_sub(to_sub_format(arg2))
		arg3 = str(input('	- name of the output file: '))
		call_Rscript(arg1, arg2, arg3)

	else:
		print('It seems like you already created the confusion matrix.')
		dat_1 = str(input('\t - filename of confusion matrix (%): '))
		dat_1 = check_existence_csv(str(dat_1) + '.csv')
		dat_1_v = str(input('\t - filename of confusion matrix with values: '))
		dat_1_v = check_existence_csv(dat_1_v + '.csv')
		submission = str(input('\t - submission (number of steps) and optional endings: '))
		submission = check_existence_sub(to_sub_format(submission))
		argf = str(input('	- filename of the folder list: '))
		argf = check_existence_csv(argf + '.csv')
		f_list = argf

	dat_2 = str(input('DataFrame by which\'s occurences to sort (new csv name or \'same\'): '))
	if (dat_2 != 'same'):
		dat_2 = check_existence_csv(dat_2 + '.csv')
	
	if (conf == 'y' or conf == 'yes'): 
		submission = to_sub_format(arg2)
	else:
		submission = to_sub_format(submission)

	colorbar_bool = yes_no_bool('Heatmap with colorbar? ')
	output_name = str(input('Name of output file: '))
	heatmap_title = str(input('Title of the heatmap: '))

	if (dat_2 == 'same') and (conf == 'y' or conf == 'yes'):
		dat_v = str(arg3) + '_val'
		x = sort_csv(arg3, submission, dat_v, heatmap_title, output_name, 1, 1, colorbar_bool)
		print('Heatmap ' + output_name + '.csv has been created.')
	elif (conf == 'y' or conf == 'yes'): # (dat_2 != 'same') 
		dat_v = str(arg3) + '_val'
		x = sort_csv_alien(dat_2, arg3, dat_v, submission, heatmap_title, output_name, 1, 1, colorbar_bool, f_list)
		print('Heatmap ' + output_name + '.csv has been created.')

	if (dat_2 == 'same') and (conf == 'n' or conf == 'no'):
		x = sort_csv(dat_1, submission, dat_1_v, heatmap_title, output_name, 1, 1, colorbar_bool)
		print('Heatmap ' + output_name + '.csv has been created.')
	elif (conf == 'n' or conf == 'no'): # (dat_2 != 'same') 
		x = sort_csv_alien(dat_2, dat_1, dat_1_v, submission, heatmap_title, output_name, 1, 1, colorbar_bool, f_list)
		print('Heatmap ' + output_name + '.csv has been created.')

	weiter = yes_no('Do you want to create more heatmaps?')

	return weiter

antwort = abfrage()

while (antwort == 'y' or antwort == 'yes'):
	plt.subplot(2,1,1)
	antwort = abfrage()
