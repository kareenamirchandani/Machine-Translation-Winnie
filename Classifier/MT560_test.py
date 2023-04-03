from datasets import load_from_disk
import csv
MT560 = load_from_disk('C:/Users/emike/MT560')

training_dictionary_list = []
training_sentences = []
training_labels = []
fields = ['sentence', 'English', 'Luganda', 'Runyankole', 'Ateso', 'Lugbara', 'Acholi','Swahili']

lugMT560 = MT560.filter(lambda ex: ex["src_lang"] == "lug")
swaMT560 = MT560.filter(lambda ex: ex["src_lang"] == "swa")
achMT560 = MT560.filter(lambda ex: ex["src_lang"] == "ach")
nynMT560 = MT560.filter(lambda ex: ex["src_lang"] == "nyn")

lug_lengthMT560 = len(lugMT560["train"])
swa_lengthMT560 = len(swaMT560["train"])
ach_lengthMT560 = len(achMT560["train"])
nyn_lengthMT560 = len(nynMT560["train"])

print(lugMT560["train"][0])
print(swaMT560["train"][0])
print(achMT560["train"][0])
print('nyn', nynMT560["train"][0])

for i in range(lug_lengthMT560):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lugMT560["train"][i]['English']}
    training_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 1, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': lugMT560["train"][i]['src']}
    training_dictionary_list.append(new_dict2)

for i in range(swa_lengthMT560):
    new_dict = {'English': 1, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':0,'sentence': swaMT560["train"][i]['English']}
    training_dictionary_list.append(new_dict)
    new_dict2 = {'English': 0, 'Luganda': 0, 'Runyankole': 0, 'Ateso': 0, 'Lugbara': 0, 'Acholi':0, 'Swahili':1,'sentence': swaMT560["train"][i]['src']}
    training_dictionary_list.append(new_dict2)

print(len(training_dictionary_list))

training_sentences_and_labels = [[] for x in range(len(training_dictionary_list))]

counter = 0
for item in training_dictionary_list:
    training_sentences.append(item['sentence'])
    training_sentences_and_labels[counter].append(item['sentence'])
    training_labels.append([item['English'], item['Luganda'], item['Runyankole'],
                   item['Ateso'], item['Lugbara'], item['Acholi'], item['Swahili']])
    training_sentences_and_labels[counter].append(item['English'])
    training_sentences_and_labels[counter].append(item['Luganda'])
    training_sentences_and_labels[counter].append(item['Runyankole'])
    training_sentences_and_labels[counter].append(item['Ateso'])
    training_sentences_and_labels[counter].append(item['Lugbara'])
    training_sentences_and_labels[counter].append(item['Acholi'])
    training_sentences_and_labels[counter].append(item['Swahili'])
    counter += 1

print('sentences and labels length',len(training_sentences_and_labels))
print('sentences length',len(training_sentences))
print('labels length',len(training_labels))

# Create csv file to store training set
with open("MT560_train.csv", "w", encoding="utf-8",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(training_sentences_and_labels)

print(training_sentences_and_labels[10])
print(training_sentences[10])
print(training_labels[10])