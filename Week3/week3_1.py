import gensim.downloader as api
import numpy as np
from numpy.linalg import norm

model = api.load("glove-twitter-100")
word1 = "beautiful"
word2 = "pretty"
print(f"cosine similarity ({word1},{word2}) with model =",model.similarity(word1,word2))
# with numpy

def simscore_with_numpy(word1,word2,model):
    vec1 = np.array(model[word1])
    vec2 = np.array(model[word2])
    simscore = np.dot(vec1,vec2)/(norm(vec1)*norm(vec2))
    return simscore

result1 = simscore_with_numpy(word1,word2,model)
print(f"cosine similarity ({word1},{word2}) with numpy =",result1)
#don't use numpy

def simscore_dont_numpy(word1,word2,model):
    vector1 = model[word1]
    vector2 = model[word2]
    dot = sum(v1*v2 for v1,v2 in zip(vector1,vector2))
    norm1 = sum(v1*v1 for v1 in vector1)**0.5
    norm2 = sum(v2*v2 for v2 in vector2)**0.5
    simiscore = dot/(norm1*norm2)
    return simiscore

result2 = simscore_dont_numpy(word1,word2,model)
print(f"cosine similarity ({word1},{word2}) dont numpy =", result2)


