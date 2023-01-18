import csv
import numpy as np

languages = []
words = []

with open("wordCount.csv", encoding="UTF8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        if len(row) < 100 and len(row) > 2:
            languages = languages + ["".join(row)]
            lang = "".join(row)
        elif len(row) > 1:
            words = words + [[lang] + row]
    wordCount = {lang: {} for lang in languages}

numLanguages = len(languages)
npLang = np.asarray(languages).reshape(1,6)

overlap = np.empty([numLanguages, numLanguages], dtype = int) ## Number of words that overlap
overlapProp = np.empty([numLanguages, numLanguages], dtype = int) ## Overlap as proportion of total number of unique words
overlapFreq = np.empty([numLanguages, numLanguages], dtype = int) ## Number of times there is an overlapping word (Columns is language which we count frequency in)
overlapFreqProp = np.empty([numLanguages, numLanguages], dtype = int) # overlapFeq as a proportion of total occurences of words in data for language
wordCountSets = {lang: set() for lang in languages}
for i in range(0, numLanguages):
    wordCountSets[languages[i]] = set(words[i*2])

for idxR, langR in enumerate(languages):
    for idxC, langC in enumerate(languages):
        if langR == langC:
            overlap[idxR, idxC] = 0
            overlapFreq[idxR, idxC] = 0
            overlapFreqProp[idxR, idxC] = 0
            overlapProp[idxR, idxC] = 0
        else:
            interSet = set.intersection(wordCountSets[langR], wordCountSets[langC])
            overlap[idxR, idxC] = len(interSet)
            overlapProp[idxR, idxC] = (float(len(interSet))/float(len(wordCountSets[langC])))*100
            count = 0
            total = sum([int(num) for num in words[idxC * 2 + 1][1:]])
            for i in interSet:
                count += int(words[idxC * 2 + 1][ words[idxC * 2].index(i)])
            overlapFreq[idxR, idxC] = count
            overlapFreqProp[idxR, idxC] = (float(count)/float(total))*100

print(languages)
print(overlap)
print("\n")
print(languages)
print(overlapProp)
print("\n")
print(languages)
print(overlapFreq)
print("\n")
print(languages)
print(overlapFreqProp)