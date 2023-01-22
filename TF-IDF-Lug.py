# Tf-idf can be successfully used for stop-words filtering from the text document
# The large word count of words like 'the', 'an' is meaningless towards the analysis of the text
# TF-IDF is a numerical statistic which measures the importance of the word in a document.
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity

# Sentence 1 means "Eggplants always grow best under warm conditions."
# Sentences 2 and 3 both mean Farmland is sometimes a challenge to farmers."
doc_1_lug = "Bbiringanya lubeerera  asinga kukulira mu mbeera ya bugumu"
doc_2_lug = "Ettaka ly'okulimirako n'okulundirako ebiseera ebimu kisoomooza abalimi"
doc_3_run = "Eitaka ry'okuhingamu, obumwe n'obumwe nirireetera abahingi oburemeezi."

data = [doc_1_lug, doc_2_lug, doc_3_run]

Tfidf_vect = TfidfVectorizer()
vector_matrix = Tfidf_vect.fit_transform(data)

tokens = Tfidf_vect.get_feature_names_out()


def create_dataframe(matrix, tokens):
    doc_names = [f'doc_{i + 1}' for i, _ in enumerate(matrix)]
    df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
    return df


output = create_dataframe(vector_matrix.toarray(), tokens)
print(tabulate(output, headers='keys'))
# This table shows that 'the' has the highest value
# TF-IDF can be used to find the most common words in a corpus

cosine_similarity_matrix = cosine_similarity(vector_matrix)
similarity = create_dataframe(cosine_similarity_matrix, ['doc_1_lug', 'doc_2_lug', 'doc_3_run'])
print(tabulate(similarity, headers='keys'))

# Unsurprisingly, we end up with similarities of 0 where there is no overlap at all

# Try again using the whole corpus of each language

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

corpus_eng = ''.join(corpus_eng)
corpus_ach = ''.join(corpus_ach)
corpus_lgg = ''.join(corpus_lgg)
corpus_lug = ''.join(corpus_lug)
corpus_run = ''.join(corpus_run)
corpus_teo = ''.join(corpus_teo)

data = [corpus_eng,corpus_ach,corpus_lgg,corpus_lug,corpus_run,corpus_teo]

Tfidf_vect = TfidfVectorizer()
vector_matrix = Tfidf_vect.fit_transform(data)

tokens = Tfidf_vect.get_feature_names_out()

def create_dataframe_salt(matrix, tokens):
    doc_names = ['corpus_eng','corpus_ach','corpus_lgg','corpus_lug','corpus_run','corpus_teo']
    df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
    return df


cosine_similarity_matrix = cosine_similarity(vector_matrix)
similarity = create_dataframe_salt(cosine_similarity_matrix, ['corpus_eng','corpus_ach','corpus_lgg','corpus_lug','corpus_run','corpus_teo'])
print(tabulate(similarity, headers='keys'))

# Similarity between English and Lugbara is greater than similarity between Acholi and Lugbara
# Results aren't exactly consistent as cosine similarity relies on there being overlap
# Where there is very little overlap, the results don't mean much
# Greatest overlap appears to be between Lugbara and Acholi
