import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

lang_counts_eng_v3 = [997, 0, 0, 0, 0, 0, 0]
lang_counts_lug_v3 = [3, 969, 25, 0, 0, 0, 0]
lang_counts_swa_v3 = [0, 0, 0, 0, 0, 0, 997]

lang_counts_eng_v2 = [997, 0, 0, 0, 0, 0, 0]
lang_counts_lug_v2 = [11, 960, 10, 5, 10, 1, 0]

lang_counts_eng_v1 = [995, 0, 0, 0, 2, 0, 0]
lang_counts_lug_v1 = [0, 973, 21, 0, 3, 0, 0]

lang_counts_eng_v4 = [997, 0, 0, 0, 0, 0, 0]
lang_counts_lug_v4 = [0, 981, 14, 0, 1, 0, 1]
lang_counts_swa_v4 = [0, 0, 0, 0, 0, 0, 997]

lang_counts_eng_v5 = [699, 296, 0, 0, 2, 0, 0]
lang_counts_lug_v5 = [195, 372, 0, 0, 5, 2, 423]
lang_counts_swa_v5 = [0, 784, 1, 0, 1, 0, 211]


'''
v3 = np.zeros((7,7))
v3[0] = lang_counts_eng_v3
v3[1] = lang_counts_lug_v3
v3[6] = lang_counts_swa_v3

ax= plt.subplot()
sns.heatmap(v3, annot=True, fmt='g', ax=ax, cmap='Greens');  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['eng', 'lug', 'nyn', 'teo', 'lgg', 'ach', 'swa']);
ax.yaxis.set_ticklabels(['eng', 'lug', 'nyn', 'teo', 'lgg', 'ach', 'swa']);
plt.show()

v2 = np.zeros((7,7))
v2[0] = lang_counts_eng_v2
v2[1] = lang_counts_lug_v2

ax= plt.subplot()
sns.heatmap(v2, annot=True, fmt='g', ax=ax, cmap='Greens');  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['eng', 'lug', 'nyn', 'teo', 'lgg', 'ach', 'swa']);
ax.yaxis.set_ticklabels(['eng', 'lug', 'nyn', 'teo', 'lgg', 'ach', 'swa']);
plt.show()

v1 = np.zeros((7,7))
v1[0] = lang_counts_eng_v1
v1[1] = lang_counts_lug_v1

ax= plt.subplot()
sns.heatmap(v1, annot=True, fmt='g', ax=ax, cmap='Greens');  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['eng', 'lug', 'nyn', 'teo', 'lgg', 'ach', 'swa']);
ax.yaxis.set_ticklabels(['eng', 'lug', 'nyn', 'teo', 'lgg', 'ach', 'swa']);
plt.show()'''

v5 = np.zeros((7,7))
v5[0] = lang_counts_eng_v5
v5[1] = lang_counts_lug_v5
v5[6] = lang_counts_swa_v5

ax= plt.subplot()
sns.heatmap(v5, annot=True, fmt='g', ax=ax, cmap='Greens');  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix: FLORES-101, Classification Model V2');
ax.xaxis.set_ticklabels(['eng', 'lug', 'nyn', 'teo', 'lgg', 'ach', 'swa']);
ax.yaxis.set_ticklabels(['eng', 'lug', 'nyn', 'teo', 'lgg', 'ach', 'swa']);
plt.savefig('cm_flores_v2.svg')
plt.show()