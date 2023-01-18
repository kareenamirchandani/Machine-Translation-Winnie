from datasets import load_dataset, concatenate_datasets
import string
import csv
dataset = load_dataset("Sunbird/salt-dataset", split="train")
# dataset = dataset[0:10]
languages = ["Luganda", "Runyankole", "Ateso", "Lugbara", "Acholi", "English"]
corpuses = {}
for lang in languages:
    corpuses[lang] = [item for sublist in dataset[lang] for item in sublist.translate(str.maketrans('', '', string.punctuation)).lower().split()]
WordCount = {key: dict.fromkeys(set(value)) for key, value in corpuses.items()}

print("Empty Word Count Created")

for lang in languages:
    for word in WordCount[lang].keys():
        WordCount[lang][word] = corpuses[lang].count(word)
    print("Added Count for " + str(lang))



print(WordCount["English"]["good"])

print("Writing to CSV")

with open('wordCount.csv', 'w', encoding="UTF8") as file:
    writer = csv.writer(file)
    for lang, wordC in WordCount.items():
        writer.writerow(lang)
        writer.writerow(wordC.keys())
        writer.writerow(wordC.values())


    