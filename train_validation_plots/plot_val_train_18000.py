import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import math

fig = plt.figure(figsize=(9,3))
#fig.suptitle('Experiment 3 training and validation accuracies')

################################## 20
train_table = '20_18000/run_train-tag-acc_greedy_1.csv'	
dat_train = pd.read_csv(train_table)  
val_table = '20_18000/run_validation-tag-acc_greedy_1.csv'
dat_val = pd.read_csv(val_table)   
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax1 = plt.subplot(1,3,1)
plt.axvline(x=9000, ymin=-0.5, ymax=1.5, color='#676663', linewidth=1.25, linestyle='--')
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 
#plt.vlines(x=9000, ymin=-0.5, ymax=1.5, color='#676663', linewidth=1.5, linestyle='--')
plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000, 12000, 15000, 18000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k', '12k', '15k', '18k']
plt.xticks(ticks, label)
ax1.tick_params(axis=u'both', which=u'both',length=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)

plt.legend(loc='lower right')
plt.title('threshold 20')

################################## 50
train_table = '50_18000/run_train-tag-acc_greedy_1.csv'	
dat_train = pd.read_csv(train_table)  
val_table = '50_18000/run_validation-tag-acc_greedy_1.csv'
dat_val = pd.read_csv(val_table)   
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax2 = plt.subplot(1,3,2, sharey=ax1)
plt.axvline(x=9000, ymin=-0.5, ymax=1.5, color='#676663', linewidth=1.25, linestyle='--')
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 
plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000, 12000, 15000, 18000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k', '12k', '15k', '18k']
plt.xticks(ticks, label)
ax2.tick_params(axis=u'both', which=u'both',length=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
plt.legend(loc='lower right')
plt.title('threshold 50')

################################## 100
train_table = '100_18000/run_train-tag-acc_greedy_1.csv'	
dat_train = pd.read_csv(train_table)  
val_table = '100_18000/run_validation-tag-acc_greedy_1.csv'
dat_val = pd.read_csv(val_table)   
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax3 = plt.subplot(1,3,3, sharey=ax1)
plt.axvline(x=9000, ymin=-0.5, ymax=1.5, color='#676663', linewidth=1.25, linestyle='--')
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 
plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000, 12000, 15000, 18000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k', '12k', '15k', '18k']
plt.xticks(ticks, label)
ax3.tick_params(axis=u'both', which=u'both',length=0)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["bottom"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
plt.legend(loc='lower right')
plt.title('threshold 100')

plt.savefig('train_val_18000.png', dpi=400)
plt.subplots_adjust(top=0.8)
plt.show()