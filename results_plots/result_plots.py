import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots()
x = [1,4,7]

# ldl vs cnn-ctc-lstm
# unbalanced 1. ldl, 2. cnn-ctc-lstm: 100, 50, 20
ldl_u = [0.118, 0.101, 0.0858]
ldl_u_r = [0.196, 0.172, 0.154]
cnn_u = [0.475, 0.366, 0.34]

labels = ['100', '50', '20']

ax.plot(x, ldl_u, linestyle='--', marker='o',color='#944de4', label='LDL normal semvecs')
ax.plot(x, ldl_u_r, linestyle='--', marker='v',color='#290057', label='LDL random semvecs')

ax.plot(x, cnn_u, linestyle='solid', marker='*',color='#ff7b00', label='CNN-CTC-LSTM')
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
ax.set_title('Unbalanced datasets')

ax.legend(loc='upper right', frameon=False)

ax.set_xlabel('threshold')
#ax.set_title('Accuracies of LDL and CNN-CTC-LSTM with unbalanced datasets')
ax.set_ylabel('proportion')
ax.set_ylim(bottom=-0.05, top=1,)
plt.xticks(x, labels)

plt.savefig('ldl_u.png', dpi=400)

fig, ax = plt.subplots()

x = [1,4,7]

# ldl vs cnn-ctc-lstm
# unbalanced 1. ldl, 2. cnn-ctc-lstm: 100, 50, 20
ldl_b = [0.0234, 0.0101, 0.00315] 
cnn_b = [0.521, 0.348, 0.253]

ldl_b_r = [0.0561, 0.0256, 0.0099]

labels = ['100', '50', '20']

ax.plot(x, ldl_b, linestyle='--', marker='o',color='#944de4', label='LDL normal semvecs')
ax.plot(x, ldl_b_r, linestyle='--', marker='v',color='#290057', label='LDL random semvecs')

ax.plot(x, cnn_b, linestyle='solid', marker='*',color='#ff7b00', label='CNN-CTC-LSTM')
ax.tick_params(axis=u'both', which=u'both',length=0)
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_facecolor('#eeeef5')
ax.legend(loc='upper right', frameon=False)
ax.set_title('Balanced datasets')

ax.set_xlabel('threshold')
#ax.set_title('Accuracies of LDL and CNN-CTC-LSTM with balanced datasets')
ax.set_ylabel('proportion')
ax.set_ylim(bottom=-0.05, top=1,)
plt.xticks(x, labels)
plt.savefig('ldl_b.png', dpi=400)


################### Experiment 4 ###################
fig, ax = plt.subplots()

x = [1,3,5,7]

exp_4_mean = [0.139, 0.118, 0.144, 0.127]
exp_4_ov = [0.341, 0.308, 0.351, 0.317]
exp_4_un = [0.455, 0.476, 0.433, 0.485]

labels = ['original', '4.1', '4.2', '4.3']
ax.axhline(y=0.139, xmin=0.0, xmax=1.0, color='#787878', linewidth=1, linestyle=':')
ax.axhline(y=0.341, xmin=0.0, xmax=1.0, color='#787878', linewidth=1, linestyle=':')
ax.axhline(y=0.455, xmin=0.0, xmax=1.0, color='#787878', linewidth=1, linestyle=':')
ax.plot(x, exp_4_mean, marker='*',color='#00ff80', label='mean accuracy')
ax.plot(x, exp_4_ov, marker='o',color='#3E8989', label='overall accuracy', linestyle='solid')
ax.plot(x, exp_4_un, marker='v',color='#DB93B0', label='unknown', linestyle='solid')

ax.set_ylim(bottom=0, top=1,)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
ax.legend(loc='upper right', frameon=False)

#ax.set_title('Results of experiment 1')
ax.set_ylabel('proportion')
#ax.set_ylim(bottom=0, top=1,)
plt.xticks(x, labels)
plt.savefig('exp4.png', dpi=400)

################### Experiment 3 ###################
fig, ax = plt.subplots()

x = [1,3,5]

# 20 50 100, 1-unknown
exp_1_mean = [0.0898, 0.139, 0.28]
exp_1_ov = [0.306, 0.341, 0.452]
exp_1_un = [1-0.457, 1-0.455, 1-0.283]

# 100 50 20
exp_1_mean = [0.28, 0.139, 0.0898]
exp_1_ov = [0.452, 0.341, 0.306]
exp_1_un = [0.283, 0.455, 0.457]

# 20 50 100, 1-unknown
exp_3_mean = [0.105, 0.159, 0.316]
exp_3_ov = [0.341, 0.366, 0.475]
exp_3_un = [1-0.434, 1-0.434, 1-0.282]

# 100 50 20
exp_3_mean = [0.316, 0.159, 0.105]
exp_3_ov = [0.475, 0.366, 0.341]
exp_3_un = [0.282, 0.434, 0.434]


labels = ['threshold 100', 'threshold 50', 'threshold 20']

ax.plot(x, exp_3_mean, marker='*',color='#00ff80', label='Experiment 3, mean accuracy', linestyle='--')
ax.plot(x, exp_1_mean, marker='*', color='#00ff80', label='Experiment 1, mean accuracy')
ax.plot(x, exp_3_ov, marker='o',color='#3E8989', label='Experiment 3, overall accuracy', linestyle='--')
ax.plot(x, exp_1_ov, marker='o', color='#3E8989', label='Experiment 1, overall accuracy')
ax.plot(x, exp_3_un, marker='v',color='#DB93B0', label='Experiment 3, unknown', linestyle='--')
ax.plot(x, exp_1_un, marker='v', color='#DB93B0', label='Experiment 1, unknown')
ax.set_ylim(bottom=0, top=1,)

ax.legend(loc='upper left', frameon=False)

#ax.set_title('Results of experiment 1 and 2')
ax.set_ylabel('proportion')
ax.set_ylim(bottom=0, top=1,)
plt.xticks(x, labels)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
plt.savefig('exp3.png', dpi=400)

################### Experiment 2 ###################
fig, ax = plt.subplots()

x = [1,3,5]

# exp_1_100 = [0.28, 0.452, 1-0.283]
# exp_1_50 = [0.139, 0.341, 1-0.455]
# exp_1_20 = [0.0898, 0.306, 1-0.457]

# exp_2_100 = [0.521, 0.524, 1-0.341]
# exp_2_50 = [0.348, 0.384, 1-0.566]
# exp_2_20 = [0.253, 0.253, 1-0.673]


exp_1_mean = [0.28, 0.139, 0.0898]
exp_1_ov = [0.452, 0.341, 0.306]
exp_1_un = [0.283, 0.455, 0.457]


exp_2_mean = [0.521, 0.348, 0.253]
exp_2_ov = [0.524, 0.384, 0.253]
exp_2_un = [0.341, 0.566, 0.673]


labels = ['threshold 100', 'threshold 50', 'threshold 20']

ax.plot(x, exp_2_mean, marker='*',color='#00ff80', label='Experiment 2, mean accuracy', linestyle='--')
ax.plot(x, exp_1_mean, marker='*',color='#00ff80', label='Experiment 1, mean accuracy')
ax.plot(x, exp_2_ov, marker='o',color='#3E8989', label='Experiment 2, overall accuracy', linestyle='--')
ax.plot(x, exp_1_ov, marker='o', color='#3E8989', label='Experiment 1, overall accuracy')
ax.plot(x, exp_2_un, marker='v',color='#DB93B0', label='Experiment 2, unknown', linestyle='--')
ax.plot(x, exp_1_un, marker='v', color='#DB93B0', label='Experiment 1, unknown')

ax.set_ylim(bottom=0, top=1,)

ax.legend(loc='upper left', frameon=False)

#ax.set_title('Results of experiment 1 and 2')
ax.set_ylabel('proportion')
ax.set_ylim(bottom=0, top=1,)
plt.xticks(x, labels)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
plt.savefig('exp2.png', dpi=400)

################### Experiment 1 ###################
fig, ax = plt.subplots()

x = [1,3,5]

exp_1_mean = [0.28, 0.139, 0.0898]
exp_1_ov = [0.452, 0.341, 0.306]
exp_1_un = [0.283, 0.455, 0.457]

labels = ['threshold 100', 'threshold 50', 'threshold 20']

ax.plot(x, exp_1_mean, marker='*',color='#00ff80', label='mean accuracy')
ax.plot(x, exp_1_ov, marker='o',color='#3E8989', label='overall accuracy')
ax.plot(x, exp_1_un, marker='v',color='#DB93B0', label='unknown')
ax.set_ylim(bottom=0, top=1,)

ax.legend(loc='upper left', frameon=False)

#ax.set_title('Results of experiment 1')
ax.set_ylabel('proportion')
#ax.set_ylim(bottom=0, top=1,)
plt.xticks(x, labels)
ax.tick_params(axis=u'both', which=u'both',length=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_facecolor('#eeeef5')
plt.grid(color='white', linewidth=2)
# only y axis grid
plt.gca().xaxis.grid(False)
plt.savefig('exp1.png', dpi=400)


# BA presentation plots

# # ldl vs cnn-ctc-lstm
# # unbalanced 1. ldl, 2. cnn-ctc-lstm: 100, 50, 20
# ldl_u = [0.118, 0.101, 0.0858]
# cnn_u = [0.475, 0.366, 0.34]

# labels = ['100', '50', '20']

# ax.plot(x, ldl_u, linestyle='--', marker='o',color='#6601d9', label='LDL')
# ax.plot(x, cnn_u, linestyle='--', marker='o',color='#00ff80', label='CNN-CTC-LSTM')
# ax.set_title('Unbalanced datasets')
# ax.legend(loc='upper right', frameon=False)

# ax.set_xlabel('threshold')
# #ax.set_title('Accuracies of LDL and CNN-CTC-LSTM with unbalanced datasets')
# ax.set_ylabel('accuracy')
# ax.set_ylim(bottom=-0.05, top=1,)
# plt.xticks(x, labels)
# plt.savefig('ldl_u.png', dpi=400)

# fig, ax = plt.subplots()

# x = [1,4,7]

# # ldl vs cnn-ctc-lstm
# # unbalanced 1. ldl, 2. cnn-ctc-lstm: 100, 50, 20
# ldl_b = [0.0234, 0.0101, 0.00315]
# cnn_b = [0.521, 0.348, 0.253]

# labels = ['100', '50', '20']

# ax.plot(x, ldl_b, linestyle='--', marker='o',color='#6601d9', label='LDL')
# ax.plot(x, cnn_b, linestyle='--', marker='o',color='#00ff80', label='CNN-CTC-LSTM')
# ax.legend(loc='upper right', frameon=False)
# ax.set_title('Balanced datasets')

# ax.set_xlabel('threshold')
# #ax.set_title('Accuracies of LDL and CNN-CTC-LSTM with balanced datasets')
# ax.set_ylabel('accuracy')
# ax.set_ylim(bottom=-0.05, top=1,)
# plt.xticks(x, labels)
# plt.savefig('ldl_b.png', dpi=400)

# ################### Experiment 4 ###################
# fig, ax = plt.subplots()

# x = [1,3,5]

# net = [0.139, 0.341, 1-0.455]
# exp_41 = [0.118, 0.308, 1-0.476]
# exp_42 = [0.144, 0.351, 1-0.433]
# exp_43 = [0.127, 0.317, 1-0.485]

# labels = ['mean accuracy', 'overall accuracy', '1-unknown']

# ax.plot(net, x, linestyle='--', marker='o',color='#6601d9', label='present network')
# ax.plot(x, exp_41, linestyle='--', marker='o',color='#00ff80', label='Experiment 4.1')
# ax.plot(x, exp_42, linestyle='--', marker='o',color='#244F26', label='Experiment 4.2')
# ax.plot(x, exp_43, linestyle='--', marker='o',color='#AAE176', label='Experiment 4.3')
# ax.legend(loc='upper left', frameon=False)

# #ax.set_title('Accuracies of different pre-prosessing approaches')
# ax.set_ylabel('accuracy')
# #ax.set_ylim(bottom=0, top=1,)
# plt.xticks(x, labels)
# plt.show()


# ################### Experiment 3 ###################
# # threshold 100
# fig, ax = plt.subplots()

# x = [1,3,5]

# exp_1_100 = [0.28, 0.452, 1-0.283]
# exp_3_100 = [0.316, 0.475, 1-0.282]
# exp_1_50 = [0.139, 0.341, 1-0.455]
# exp_3_50 = [0.159, 0.366, 1-0.434]
# exp_1_20 = [0.0898, 0.306, 1-0.457]
# exp_3_20 = [0.105, 0.341, 1-0.434]

# labels = ['mean accuracy', 'overall accuracy', '1-unknown']

# ax.plot(x, exp_1_100, marker='o',color='#00ff80', label='Experiment 1, threshold 100')
# ax.plot(x, exp_3_100, linestyle=':', marker='o',color='#00ff80', label='Experiment 3, threshold 100')
# ax.plot(x, exp_1_50, marker='o',color='#3E8989', label='Experiment 1, threshold 50')
# ax.plot(x, exp_3_50, linestyle=':',marker='o',color='#3E8989', label='Experiment 3, threshold 50')
# ax.plot(x, exp_1_20, marker='o',color='#DB93B0', label='Experiment 1, threshold 20')
# ax.plot(x, exp_3_20, linestyle=':', marker='o',color='#DB93B0', label='Experiment 3, threshold 20')

# ax.legend(loc='upper left', frameon=False)

# #ax.set_title('Results of experiment 1 and 3')
# ax.set_ylabel('accuracy')
# #ax.set_ylim(bottom=0, top=1,)
# plt.xticks(x, labels)
# plt.show()

# ################### Experiment 2 ###################

# fig, ax = plt.subplots()

# x = [1,3,5]

# exp_1_100 = [0.28, 0.452, 1-0.283]
# exp_2_100 = [0.521, 0.524, 1-0.341]
# exp_1_50 = [0.139, 0.341, 1-0.455]
# exp_2_50 = [0.348, 0.384, 1-0.566]
# exp_1_20 = [0.0898, 0.306, 1-0.457]
# exp_2_20 = [0.253, 0.253, 1-0.673]

# labels = ['mean accuracy', 'overall accuracy', '1-unknown']

# ax.plot(x, exp_1_100, marker='o',color='#00ff80', label='Experiment 1, threshold 100')
# ax.plot(x, exp_2_100, linestyle=':', marker='o',color='#00ff80', label='Experiment 2, threshold 100')
# ax.plot(x, exp_1_50, marker='o',color='#3E8989', label='Experiment 1, threshold 50')
# ax.plot(x, exp_2_50, linestyle=':',marker='o',color='#3E8989', label='Experiment 2, threshold 50')
# ax.plot(x, exp_1_20, marker='o',color='#DB93B0', label='Experiment 1, threshold 20')
# ax.plot(x, exp_2_20, linestyle=':', marker='o',color='#DB93B0', label='Experiment 2, threshold 20')

# ax.legend(loc='upper left', frameon=False)

# #ax.set_title('Results of experiment 1 and 2')
# ax.set_ylabel('accuracy')
# ax.set_ylim(bottom=0, top=1,)
# plt.xticks(x, labels)
# plt.show()

# ################### Experiment 1 ###################
# fig, ax = plt.subplots()

# x = [1,3,5]

# exp_1_100 = [0.28, 0.452, 1-0.283]
# exp_1_50 = [0.139, 0.341, 1-0.455]
# exp_1_20 = [0.0898, 0.306, 1-0.457]

# labels = ['mean accuracy', 'overall accuracy', '1-unknown']

# ax.plot(x, exp_1_100, marker='o',color='#00ff80', label='Experiment 1, threshold 100')
# ax.plot(x, exp_1_50, marker='o',color='#3E8989', label='Experiment 1, threshold 50')
# ax.plot(x, exp_1_20, marker='o',color='#DB93B0', label='Experiment 1, threshold 20')


# ax.legend(loc='upper left', frameon=False)

# #ax.set_title('Results of experiment 1')
# ax.set_ylabel('accuracy')
# #ax.set_ylim(bottom=0, top=1,)
# plt.xticks(x, labels)
# plt.show()