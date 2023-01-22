# https://www.analyticsvidhya.com/blog/2021/05/measuring-text-similarity-using-bert/

# The following code uses a sentence-transformers model
# It maps sentences and paragraphs to a 768 dimensional dense vector space,
# and it can be used for tasks like clustering or semantic search

# This model is depreciated

# Write some lines to encode
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sen = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell.",
    "3  years  later ,  the  coffin  was  still  full  of  Jello.",
]

model = SentenceTransformer('bert-base-nli-mean-tokens')

# Encoding:
sen_embeddings = model.encode(sen)
sen_embeddings.shape

# Calculate cosine similarity for sentence 0:
cosine_similarity(
    [sen_embeddings[0]],
    sen_embeddings[1:]
)

print(cosine_similarity(
    [sen_embeddings[0]],
    sen_embeddings[1:]
))

# The first and last sentences are exactly the same so cosine similarity = 1
# As a test, I inserted extra spaces into the 'copy' sentence. It didn't affect the similarity
# Changing 'Three' to '3' brought the similarity down to 0.9904
