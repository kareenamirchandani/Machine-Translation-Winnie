# Tf-idf can be successfully used for stop-words filtering from the text document
# The large word count of words like 'the', 'an' is meaningless towards the analysis of the text
# TF-IDF is a numerical statistic which measures the importance of the word in a document.

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity

doc_1 = "Data is the oil of the digital economy"
doc_2 = "Data is a new oil"

data = [doc_1, doc_2]

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
similarity = create_dataframe(cosine_similarity_matrix,['doc_1','doc_2'])
print(tabulate(similarity, headers='keys'))

# So, using TF-IDF, the cosine similarity between doc_1 and doc_2 is 0.3279
# Again, the word 'a' disappeared...
