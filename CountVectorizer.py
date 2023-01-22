from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity

# Define the sample text documents and apply the CountVectorizer to them
# The CountVectorizer is used to count the occurrence of each word in the document

doc_1 = "Data is the oil of the digital economy"
doc_2 = "Data is a new oil"

data = [doc_1, doc_2]
count_vectorizer = CountVectorizer()

vector_matrix = count_vectorizer.fit_transform(data)

print(vector_matrix)

# The generated vector matrix is a sparse matrix,
# that is not printed here.
# Convert it to numpy array and display it with the token word.

tokens = count_vectorizer.get_feature_names_out()
print(tokens)

# Convert to numpy array
vector_matrix = vector_matrix.toarray()
print(vector_matrix)


# For some reason, 'a' didn't make it into the list of tokens.
# Does CountVectorize remove stopwords?

# Create pandas dataframe to display tokens and counts

def create_dataframe(matrix, tokens):
    doc_names = [f'doc_{i + 1}' for i, _ in enumerate(matrix)]
    df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
    return df


counts = create_dataframe(vector_matrix, tokens)
print(tabulate(counts, headers='keys'))

# Compute the cosine similarity between doc_1 and doc_2

cosine_similarity_matrix = cosine_similarity(vector_matrix)
similarity = create_dataframe(cosine_similarity_matrix,['doc_1','doc_2'])
print(tabulate(similarity, headers='keys'))

# By observing the table, we can see there is a similarity of 0.47 between doc_1 and doc_2

