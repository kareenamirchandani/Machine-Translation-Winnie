import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
lang_counts_eng_v3 = [99190, 466, 13, 5, 2, 14, 310]
lang_counts_lug_v3 = [32284, 63298, 875, 2, 5, 7, 3529]
lang_counts_nyn_v3 = [40656, 12339, 40767, 49, 15, 6, 6168]
lang_counts_teo_v3 = [20998, 418, 99, 17360, 6, 13, 1402]
lang_counts_lgg_v3 = [30722, 4487, 925, 171, 20388, 522, 3022]
lang_counts_ach_v3 = [34517, 6203, 861, 687, 56, 38637, 19039]
lang_counts_swa_v3 = [24924, 375, 26, 7, 4, 42, 74622]

lang_counts_eng_v3 = [5000, 0, 0, 0, 0, 0, 0]
lang_counts_lug_v3 = [18, 4931, 41, 0, 2, 0, 8]
lang_counts_nyn_v3 = [39, 61, 4893, 0, 1, 0, 6]
lang_counts_teo_v3 = [16, 0, 8, 4974, 1, 1, 0]
lang_counts_lgg_v3 = [6, 6, 3, 2, 4979, 2, 2]
lang_counts_ach_v3 = [6, 4, 1, 1, 4, 4979, 5]
lang_counts_swa_v3 = [11, 1, 0, 0, 0, 0, 4988]

lang_counts_eng_v3 = [99298, 31, 8, 5, 12, 136, 510]
lang_counts_lug_v3 = [18644, 67266, 310, 5, 441, 546, 12788]
lang_counts_nyn_v3 = [21642, 12143, 45324, 76, 420, 770, 19625]
lang_counts_teo_v3 = [854, 317, 64, 30263, 103, 436, 8259]
lang_counts_lgg_v3 = [10778, 3165, 649, 138, 31643, 5233, 8631]
lang_counts_ach_v3 = [16961, 772, 133, 116, 248, 75469, 6301]
lang_counts_swa_v3 = [18888, 110, 7, 24, 18, 146, 80807]'''

lang_counts_eng_v3 = [4081, 851, 20, 0, 22, 26, 0]
lang_counts_lug_v3 = [1010, 3206, 16, 1, 70, 42, 655]
lang_counts_nyn_v3 = [35, 191, 1889, 0, 262, 25, 2598]
lang_counts_teo_v3 = [7, 2152, 1, 1436, 44, 1111, 249]
lang_counts_lgg_v3 = [17, 22, 1, 1, 4498, 314, 147]
lang_counts_ach_v3 = [46, 12, 2, 0, 3, 4910, 27]
lang_counts_swa_v3 = [10, 2773, 24, 0, 5, 0, 2188]

lang_counts_eng_v3= np.array(lang_counts_eng_v3,dtype=float)
lang_counts_eng_v3 = np.round(lang_counts_eng_v3/5000*100,2)

lang_counts_lug_v3= np.array(lang_counts_lug_v3,dtype=float)
lang_counts_lug_v3 = np.round(lang_counts_lug_v3/5000*100,2)

lang_counts_nyn_v3= np.array(lang_counts_nyn_v3,dtype=float)
lang_counts_nyn_v3 = np.round(lang_counts_nyn_v3/5000*100,2)

lang_counts_teo_v3= np.array(lang_counts_teo_v3,dtype=float)
lang_counts_teo_v3 = np.round(lang_counts_teo_v3/5000*100,2)

lang_counts_lgg_v3= np.array(lang_counts_lgg_v3,dtype=float)
lang_counts_lgg_v3 = np.round(lang_counts_lgg_v3/5000*100,2)

lang_counts_ach_v3= np.array(lang_counts_ach_v3,dtype=float)
lang_counts_ach_v3 = np.round(lang_counts_ach_v3/5000*100,2)

lang_counts_swa_v3= np.array(lang_counts_swa_v3,dtype=float)
lang_counts_swa_v3 = np.round(lang_counts_swa_v3/5000*100,2)



v3 = np.zeros((7,7))
v3[0] = lang_counts_eng_v3
v3[1] = lang_counts_lug_v3
v3[2] = lang_counts_nyn_v3
v3[3] = lang_counts_teo_v3
v3[4] = lang_counts_lgg_v3
v3[5] = lang_counts_ach_v3
v3[6] = lang_counts_swa_v3

ax= plt.subplot()
sns.heatmap(v3, annot=True, fmt='g', ax=ax, cmap='Oranges');  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix: SALT and MT560 Sentences V2');
ax.xaxis.set_ticklabels(['eng', 'lug', 'nyn', 'teo', 'lgg', 'ach', 'swa']);
ax.yaxis.set_ticklabels(['eng', 'lug', 'nyn', 'teo', 'lgg', 'ach', 'swa']);
plt.savefig('cm_salt_mt560_sentences_v2.svg')
plt.show()
