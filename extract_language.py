import csv

with open('sunbirdData_new.csv', encoding='UTF8') as csvfile:
    reader = csv.DictReader(csvfile)

    corpus_eng = []
    corpus_ach = []
    corpus_lgg = []
    corpus_lug = []
    corpus_run = []
    corpus_teo = []

    for row in reader:
        corpus_eng.append(row['English'])
        corpus_ach.append(row['Acholi'])
        corpus_lgg.append(row['Lugbara'])
        corpus_lug.append(row['Luganda'])
        corpus_run.append(row['Runyankole'])
        corpus_teo.append(row['Ateso'])

with open('corpus_eng.txt','w') as file:
    for item in corpus_eng:
        file.write(item + '\n')

with open('corpus_ach.txt','w') as file:
    for item in corpus_ach:
        file.write(item + '\n')

with open('corpus_lgg.txt','w') as file:
    for item in corpus_lgg:
        file.write(item + '\n')

with open('corpus_lug.txt','w') as file:
    for item in corpus_lug:
        file.write(item + '\n')

with open('corpus_run.txt','w') as file:
    for item in corpus_run:
        file.write(item + '\n')

with open('corpus_teo.txt','w') as file:
    for item in corpus_teo:
        file.write(item + '\n')

# Not sure what the \ua78 character that appears a lot is