import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import math
plt.rcParams.update({'font.size': 11})
fig = plt.figure(figsize=(10,5))
#fig.suptitle('model accuracy in experiment 4.1')

################################## unbalanced
train_table = 'unbalanced_new/run_train-tag-acc_greedy_1.csv'	
dat_train = pd.read_csv(train_table)  
val_table = 'unbalanced_new/run_validation-tag-acc_greedy_1.csv'
dat_val = pd.read_csv(val_table)   
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax1 = plt.subplot(2,4,1)
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 
plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k']
plt.xticks(ticks, label)
plt.legend(loc='lower right')
plt.setp( ax1.get_xticklabels(), visible=False)
plt.title('original model \n accuracy')
#plt.set_size_inches(10, 4, forward=True)
ax1.tick_params(axis=u'both', which=u'both',length=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)


train_table = 'unbalanced_new/run_train-tag-ctc_loss.csv'	
dat_train = pd.read_csv(train_table)  
val_table = 'unbalanced_new/run_validation-tag-ctc_loss.csv'
dat_val = pd.read_csv(val_table)    
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax2 = plt.subplot(2,4,5, sharex=ax1)
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 

plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k']
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
plt.legend(loc='upper right')

plt.title('ctc loss')

################################## 4.1
train_table = 'pow_4/run_train-tag-acc_greedy_1.csv'	
dat_train = pd.read_csv(train_table)  
val_table = 'pow_4/run_validation-tag-acc_greedy_1.csv'
dat_val = pd.read_csv(val_table)   
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax1 = plt.subplot(2,4,2, sharey=ax1)
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 
plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k']
plt.xticks(ticks, label)
plt.legend(loc='lower right')
plt.setp( ax1.get_xticklabels(), visible=False)
plt.title('experiment 4.1 \n accuracy')
ax1.tick_params(axis=u'both', which=u'both',length=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
#plt.set_size_inches(10, 4, forward=True)



train_table = 'pow_4/run_train-tag-ctc_loss.csv'	
dat_train = pd.read_csv(train_table)  
val_table = 'pow_4/run_validation-tag-ctc_loss.csv'
dat_val = pd.read_csv(val_table)    
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax3 = plt.subplot(2,4,6, sharex=ax1, sharey=ax2)
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 

plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k']
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
plt.legend(loc='upper right')

plt.title('ctc loss')

################################## 4.2
train_table = 'pow_10/run_train-tag-acc_greedy_1.csv'	
dat_train = pd.read_csv(train_table)  
val_table = 'pow_10/run_validation-tag-acc_greedy_1.csv'
dat_val = pd.read_csv(val_table)   
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax1 = plt.subplot(2,4,3, sharey=ax1)
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 
plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k']
plt.xticks(ticks, label)
plt.legend(loc='lower right')
plt.setp( ax1.get_xticklabels(), visible=False)
ax1.tick_params(axis=u'both', which=u'both',length=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
plt.title('experiment 4.2 \n accuracy')
#plt.set_size_inches(10, 4, forward=True)



train_table = 'pow_10/run_train-tag-ctc_loss.csv'	
dat_train = pd.read_csv(train_table)  
val_table = 'pow_10/run_validation-tag-ctc_loss.csv'
dat_val = pd.read_csv(val_table)    
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax4 = plt.subplot(2,4,7, sharex=ax1, sharey=ax2)
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 

plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k']
plt.xticks(ticks, label)
ax4.tick_params(axis=u'both', which=u'both',length=0)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["bottom"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
plt.legend(loc='upper right')

plt.title('ctc loss')

################################## 4.3
train_table = 'pow_15/run_train-tag-acc_greedy_1.csv'	
dat_train = pd.read_csv(train_table)  
val_table = 'pow_15/run_validation-tag-acc_greedy_1.csv'
dat_val = pd.read_csv(val_table)   
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax1 = plt.subplot(2,4,4, sharey=ax1)
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 
plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k']
plt.xticks(ticks, label)
plt.legend(loc='lower right')
plt.setp( ax1.get_xticklabels(), visible=False)
plt.title('experiment 4.3 \n accuracy')
ax1.tick_params(axis=u'both', which=u'both',length=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
#plt.set_size_inches(10, 4, forward=True)



train_table = 'pow_15/run_train-tag-ctc_loss.csv'	
dat_train = pd.read_csv(train_table)  
val_table = 'pow_15/run_validation-tag-ctc_loss.csv'
dat_val = pd.read_csv(val_table)    
dat_train = dat_train.drop(['Wall time'], axis = 1)
dat_val = dat_val.drop(['Wall time'], axis = 1)

ax5 = plt.subplot(2,4,8, sharex=ax1, sharey=ax2)
plt.plot(dat_train['Step'].tolist(), dat_train['Value'].tolist(), color = '#005DBB', label='train', linewidth=0.5)
plt.plot(dat_val['Step'].tolist(), dat_val['Value'].tolist(), color = '#f06732', label='validation', linewidth=0.75) 

plt.grid(color='gainsboro', linestyle='-')
ticks = range(0, 10000, 1000)
ticks = [0, 3000, 6000, 9000]
label = ['0', '1000k', '2000k', '3000k', '4000k', '5000k', '6000k', '7000k', '8000k', '9000k']
label = ['0', '3k', '6k', '9k']
plt.xticks(ticks, label)
plt.legend(loc='upper right')

plt.title('ctc loss')
plt.subplots_adjust(wspace=0.5, right=0.975, left=0.05)
ax5.tick_params(axis=u'both', which=u'both',length=0)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.spines["bottom"].set_visible(False)
ax5.spines["left"].set_visible(False)
ax5.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
plt.savefig('train_val.png', dpi=400)
plt.show()

